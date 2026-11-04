# TaskPilot Agent Platform v3.2

> 依据《TaskPilot Agent Platform — Agent 工业级升级实战教程》构建的可运行工程。
> 代码基线：`taskpilot-industrial` v2.0.0-industrial；本目录在其上应用了 v3 审查中的 P0 修复，并接入真实 Qwen 模型。
> 定位不变：**单机、SQLite、同步执行的 Agent 可靠性工程原型**，不是生产级分布式系统。

---

## 1. 这个版本做了什么

| 项 | 状态 |
|---|---|
| 有界 ReAct 循环（`react_decide ⇄ tool_execution`） | 已实现 |
| 五类预算（步数/工具/模型/Token/绝对 deadline） | 已实现 |
| 分类重试 + `Retry-After` | 已实现 |
| 三态熔断（CLOSED/OPEN/HALF_OPEN，单探针） | 已实现 |
| SQLite 工具幂等（认领/参数摘要/租约/对账/重放） | 已实现 |
| LangGraph `SqliteSaver` 检查点 + `/resume` | 已实现 |
| 审批状态机 + 原子迁移 | 已实现 |
| Memory 分层（short_term/long_term，TTL/去重） | 已实现 |
| JSON 日志 + request/task/thread ID + Trace + 指标 | 已实现（Trace 缺节点耗时） |
| **真实 Qwen 模型接入** | **已跑通（见第 5 节）** |
| **P0：跨用户 `Idempotency-Key` 隔离** | **已修复** |
| **P0：`GET /tasks` 按用户过滤** | **已修复** |
| **P0：`/resume` 原子领取** | **已修复** |
| **P0：`mark_succeeded` 失败不再报成功** | **已修复** |
| **自由对话式多 Agent（Supervisor + 4 Worker + 共享 transcript）** | **已实现（v3.2）** |
| **对话持久化、查询、按用户列表** | **已实现** |
| 任务队列 / 租约 / 心跳 / 运行中取消 | 未实现（v3.3 计划） |
| 真实模型效果评测 / 多租户 RBAC | 未实现 |

---

## 2. 快速开始

```powershell
cd taskpilot-platform-v3
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 复制配置模板，填入自己的 Qwen Key
Copy-Item .env.example .env
# 编辑 .env：QWEN_API_KEY=...，MOCK_LLM=0 使用真实模型；MOCK_LLM=1 离线演示

# 离线测试（不需要真实模型）
python -m pytest -o addopts="" -q

# 启动 API
uvicorn app.main:app --reload
# http://127.0.0.1:8000/docs
```

> 本目录的 `.env` 已按当前环境配置好真实 Qwen Key（不提交、不打包）。  
> 交付压缩包**不包含** `.env`；收到后请使用自己的 Key。

### 真实模型配置

```dotenv
QWEN_API_KEY=你的 key
QWEN_BASE_URL=https://maas.qianwenaiapi.com/compatible-mode/v1
QWEN_CHAT_MODEL=qwen3.8-flash
MOCK_LLM=0
```

### 离线演示

```dotenv
MOCK_LLM=1
```

测试通过 `tests/conftest.py` 强制 `MOCK_LLM=1`，因此 `pytest` 不会调用真实 API。

---

## 3. 一次真实任务

```powershell
curl.exe -X POST http://127.0.0.1:8000/tasks `
  -H "Content-Type: application/json" `
  -d '{"goal":"分析 sample_predictions.csv 的 MAE、RMSE 和 R2","task_type":"data_analysis","input_file":"sample_predictions.csv"}'
```

返回中重点看：

- `status`：`completed` / `partial`
- `trace`：应有两次 `react_decide`、一次 `tool_execution`
- `observations`：工具结果
- `budget`：步数/工具/模型调用/Token
- `answer.metrics`：MAE/RMSE/R2

审批路径：

```powershell
# 1) 创建外部动作 → pending_approval
curl.exe -X POST http://127.0.0.1:8000/tasks -H "Content-Type: application/json" `
  -d '{"goal":"请发送邮件给客户，说明项目延期","user_id":"u1"}'

# 2) 批准（同一 user_id）
curl.exe -X POST http://127.0.0.1:8000/tasks/<task_id>/approve -H "Content-Type: application/json" `
  -d '{"user_id":"u1","approved":true,"reason":"确认发送范围"}'
```

### 3.2 自由对话式多 Agent（v3.2 新增）

`/tasks` 是“单图多角色”；`/conversations` 是“Supervisor + 独立 Worker + 共享对话”：

```text
用户目标 → 共享 transcript → Supervisor 决定下一位
        → research / data / writing / general → 回写 transcript
        → Supervisor 再决定 ... → finish → writing 生成最终答复
```

```powershell
curl.exe -X POST http://127.0.0.1:8000/conversations `
  -H "Content-Type: application/json" `
  -d '{"goal":"分析 sample_predictions.csv 的 MAE、RMSE 和 R2，并写一段结论","user_id":"u1","max_turns":6}'
```

返回重点：

- `handoffs`：每次“谁在什么理由下接手”；
- `messages`：user / supervisor / agent 的共享 transcript；
- `observations`：工具结果；
- `final_answer`：Writing Agent 综合后的答复；
- `agents`：当前可调度 Agent 目录（research/data/writing/general）。

设计说明见 `docs/v3/13_多Agent对话运行时.md`；`/tasks` 保留不动，两条路径并存。

---

## 4. v3 P0 修复说明

| 问题 | 修复 | 位置 | 回归测试 |
|---|---|---|---|
| 跨用户复用 `Idempotency-Key` 返回他人任务 | `task_requests` 改为 `(user_id, request_key)` 复合主键；查询/注册都带 user；旧表自动迁移到 `local-demo` | `app/services/repository.py`、`app/main.py` | `test_same_idempotency_key_is_scoped_per_user`、`test_legacy_task_requests_table_is_migrated` |
| `GET /tasks` 返回全部用户 | `list_tasks(user_id=...)`；API 传当前 user | `repository.py`、`main.py` | `test_list_tasks_is_scoped_per_user` |
| `/resume` 先读后写，两个请求都能运行 | `claim_for_resume()` 条件 UPDATE；`rowcount==1` 才 invoke；失败 409 | `repository.py`、`main.py` | `test_concurrent_resume_claim_has_one_winner` |
| 工具成功但 `mark_succeeded` 失败仍返回成功 | 返回 `IDEMPOTENCY_RECORD_FAILED`，`data.external_effect_applied=true`；不再假装成功 | `app/tools/registry.py`、`app/core/errors.py` | `test_mark_succeeded_failure_does_not_report_success` |
| Planner 编造工具名 | 生成计划后按 registry 白名单过滤；没有合法工具时回退到允许集合 | `app/agents/agent_graph.py` | 真实 smoke 中 `plan.tools` 只含 registry 工具 |

> `/resume` 现在只接受 `interrupted` / `failed`。运行中的任务需要先由 reaper 标记为 `interrupted`；这是为了避免“恢复一个还活着的任务”。

---

## 5. 验证结果

### 5.1 自动化测试

```powershell
python -m pytest -o addopts="" -q
# 76 passed in 22.64s（2026-11-04，Windows + Python 3.12.14；多次运行 19–25s）
```

证据：`outputs/test-results-v3.xml`。  
覆盖：原 64 个测试 + 6 个 P0 回归 + 6 个多 Agent 对话测试。

### 5.2 真实 Qwen smoke

```powershell
$env:MOCK_LLM="0"
python scripts/test_llm.py          # 结构化 TaskPlan 单调用
python scripts/real_e2e_smoke.py    # 完整图：planner → react → tool → writer → review
python scripts/conversation_smoke.py # 自由对话多 Agent：data → writing → finish
```

本次实际结果（2026-11-04）：

- 模型：`qwen3.8-flash`，Base URL：`https://maas.qianwenaiapi.com/compatible-mode/v1`
- `scripts/test_llm.py`：成功返回合法 `TaskPlan`，熔断器 `CLOSED`，`total_successes=1`
- `scripts/real_e2e_smoke.py`：`HTTP 200`，`status=completed`
- Trace：`classify → context_retrieval → planner → react_decide → tool_execution → react_decide → writer → review → persist`
- 真实模型调用：4 次；工具：1 次（`analyze_csv`）；Token：7403；任务耗时约 46s（多次 smoke 在 7.0k–8.1k tokens、38–53s 波动）
- 答案指标：MAE=1.35、RMSE=1.4089、R2=0.984876

证据：`outputs/real_llm_smoke.txt`、`outputs/real_e2e_smoke.json`、`outputs/real_e2e_smoke.txt`。

自由对话多 Agent smoke：

- `scripts/conversation_smoke.py`：`status=completed`，`agents=[data, writing]`
- handoff：`data → writing → finish`
- 工具：`analyze_csv` 1 次；模型调用 6 次；Token 8605；约 34s
- 最终答复包含 MAE/RMSE/R2，并主动说明样本量仅 8 条

证据：`outputs/conversation_real_smoke.json`、`outputs/conversation_real_smoke.txt`。

> 这是一次 smoke，不是质量评测；样本量 1，不能据此宣称“准确率 X%”。

---

## 6. 目录结构

```text
taskpilot-platform-v3/
├─ app/
│  ├─ main.py                  # API、认证、限流、审批、恢复、/conversations
│  ├─ agents/                  # LangGraph ReAct 图 + conversation 多 Agent 运行时
│  ├─ core/                    # 预算、重试、熔断、错误、指标、日志
│  ├─ services/                # 任务库、幂等、检查点、记忆、LLM、对话存储
│  └─ tools/                   # ToolRegistry、安全工具
├─ tests/                      # 76 个测试（含 6 个 P0 回归 + 6 个多 Agent 对话）
├─ scripts/
│  ├─ test_llm.py              # 真实模型结构化 smoke
│  ├─ real_e2e_smoke.py        # 真实模型单图端到端 smoke
│  ├─ conversation_smoke.py    # 真实模型多 Agent 对话 smoke
│  └─ crash_runner.py          # 子进程崩溃恢复
├─ docs/                       # 原有架构/审计/测试文档
├─ data/                       # SQLite（运行后生成，已 gitignore）
├─ outputs/                    # 测试报告/真实 smoke 证据
├─ requirements.txt
├─ .env.example
└─ README.md
```

---

## 7. 已知限制（不要写成“生产级”）

1. **单机 SQLite、同步执行**：没有任务队列；`POST /tasks` 返回时任务已结束。
2. **没有任务租约/心跳/运行中取消**：`/resume` 原子领取已修，但多 worker 任务级单写者仍未实现。
3. **真实模型未做质量评测**：只有 1 次 smoke；没有固定评测集/rubric/回归。
4. **Trace 缺节点耗时、`model_call_id`、任务级 Token 归因**。
5. **`user_id` 仍由请求传入**：不是认证；多租户/RBAC 未实现。
6. **Docker/MCP/Gradio 未在本环境验证**。
7. **线程式工具超时是“放弃等待”**，不能强杀阻塞线程。
8. **熔断跨进程共享**有实现，但测试不是双进程。
9. **多 Agent 对话没有恢复/取消**：`/conversations` 同步执行；进程崩溃后只能重建，未接 checkpoint。
10. **不是自治 Agent 社会**：Worker 不直接私聊，Supervisor 是单点调度；误路由只有 turn 上限兜底。
11. **多 Agent 只跑了 1 次真实 smoke**：不能证明“多 Agent 一定优于单 Agent”，需要固定评测集。

---

## 8. 关键文档

- `docs/架构说明.md`：组件/时序/故障路径
- `docs/审计报告.md`：v2 基线 P0–P3
- `docs/测试报告.md`：原 64 个测试说明
- `docs/v3/13_多Agent对话运行时.md`：Supervisor/Worker/共享对话设计与边界
- 上级工作区 `docs/v3/`：v3.0 教程、审查报告、P0/P1/P2、面试题、路线图

---

## 9. 安全

- `.env` 已在 `.gitignore` 中；不要提交 Key。
- `TASKPILOT_API_TOKEN` 为空时 API 不认证；生产必须接 JWT/网关。
- 工具白名单/参数 Schema/审批/路径限制是基础控制，不是完整安全体系。
- 日志/Trace 不打印完整 prompt 与 Key；`redact()` 是 best-effort。

---

## 10. 复现命令（最短）

```powershell
cd taskpilot-platform-v3
python -m pytest -o addopts="" -q
$env:MOCK_LLM="0"
python scripts/test_llm.py
python scripts/real_e2e_smoke.py
python scripts/conversation_smoke.py
uvicorn app.main:app --reload
```




