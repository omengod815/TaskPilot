# TaskPilot v3.3 构建报告

> 构建日期：2026-11-04
> 目标：根据《TaskPilot Agent Platform — Agent 工业级升级实战教程》构建一个可运行项目
> 新目录：`taskpilot-platform-v3/`
> 代码基线：`taskpilot-industrial/` v2.0.0-industrial
> 真实模型：Qwen `qwen3.8-flash`（Key 来自 `D:\research-rag-agent\.env`，未写入报告/未打包）

---

## 1. 构建结果

| 项 | 结果 |
|---|---|
| 项目目录 | `taskpilot-platform-v3/` |
| 自动化测试 | **77 passed，0 failed，21.82s**（2026-11-05；多次运行 19–26s） |
| 真实模型结构化 smoke | **成功**：返回合法 `TaskPlan`，熔断器 CLOSED |
| 真实模型端到端 smoke | **成功**：`status=completed`，调用 `analyze_csv`，4 次模型调用 |
| 真实任务指标 | MAE=1.35、RMSE=1.4089、R2=0.984876（来自工具结果） |
| 真实 smoke Token | 7403（多次运行 7.0k–8.1k 波动） |
| 真实 smoke 耗时 | 约 46s（多次运行 38–53s 波动） |
| 真实多 Agent 对话 smoke | **成功**：`data → writing → finish`，6 次模型调用，8605 tokens，约 34s |
| `.env` | 已按当前 Qwen 配置生成；被 `.gitignore` 忽略，**不进入交付压缩包** |
| 业务源码是否重写 | 否；v2 基线 + 4 个 P0 修复 + 2 个工程修复 + v3.2 多 Agent 运行时 + v3.3 多轮追问 |

### 新增/修改的关键文件

| 文件 | 变更 |
|---|---|
| `app/services/repository.py` | `task_requests` 复合主键迁移；`list_tasks` 按 user；`claim_for_resume` 原子领取 |
| `app/main.py` | 请求幂等按 user；`GET /tasks` 按 user；`/resume` 条件领取；`/conversations/{id}/messages` 多轮追问 |
| `app/tools/registry.py` | `mark_succeeded` 失败返回 `IDEMPOTENCY_RECORD_FAILED`，不再假装成功 |
| `app/core/errors.py` | 新增 `IDEMPOTENCY_RECORD_FAILED` |
| `app/agents/agent_graph.py` | Planner 生成的工具名按 registry 白名单过滤 |
| `scripts/test_llm.py` | 修复直接运行时的 `ModuleNotFoundError`（加入 project root） |
| `scripts/real_e2e_smoke.py` | 新增真实模型端到端 smoke（fresh user/session，强制工具调用） |
| `tests/test_v3_p0_fixes.py` | 新增 6 个 P0/迁移回归测试 |
| `app/agents/conversation/` | 新增 Supervisor + 4 个独立 Worker + 共享 transcript 运行时 |
| `app/services/conversation_store.py` | 新增对话持久化（conversations 表） |
| `app/schemas/conversations.py` | 新增 /conversations 请求/响应 Schema |
| `tests/test_conversation_multi_agent.py` | 新增 7 个多 Agent 对话测试（含多轮追问） |
| `scripts/conversation_smoke.py` | 新增真实模型多 Agent 对话 smoke |
| `README.md` | 改写为 v3.3 使用说明与边界 |

---

## 2. 为什么应用 P0 修复

v3 审查（`docs/v3/01_技术审查报告.md`）用探针复现了两个越权问题，并确认了两个一致性窗口。直接照搬 v2 代码会把已知缺陷带进“新项目”，因此本次构建先修：

| 问题 | 修复前 | 修复后 | 回归测试 |
|---|---|---|---|
| 跨用户 `Idempotency-Key` | Bob 用 Alice 的 key 拿到 Alice 的任务 | 复合主键 `(user_id, request_key)`；不同用户可复用同一 key，互不可见 | `test_same_idempotency_key_is_scoped_per_user` |
| `GET /tasks` 越权 | 返回全部用户任务 | `list_tasks(user_id=...)`；API 传当前 user | `test_list_tasks_is_scoped_per_user` |
| `/resume` 并发 | 两个请求都能读 `interrupted` 并 invoke | `claim_for_resume()` 条件 UPDATE，`rowcount==1` 才执行；失败 409 | `test_concurrent_resume_claim_has_one_winner` |
| `mark_succeeded` 失败 | 吞异常，仍返回 `ok=True` | 返回 `ok=False` + `IDEMPOTENCY_RECORD_FAILED`；`data.external_effect_applied=true` | `test_mark_succeeded_failure_does_not_report_success` |
| 旧库迁移 | 旧 `task_requests` 只有全局 key | 自动重建为复合主键，旧行归到 `local-demo` | `test_legacy_task_requests_table_is_migrated` |
| Planner 编造工具 | 计划里出现 `pandas/matplotlib` 等未注册工具 | 按 registry 白名单过滤；无合法工具时回退允许集合 | 真实 smoke 观察 |

> `/resume` 现在只接受 `interrupted` / `failed`。运行中的任务须先被 reaper 标为 `interrupted`，避免“恢复一个还活着的任务”。任务租约/运行中心跳仍未实现（v3.2）。

---

## 3. 测试证据

### 3.1 命令

```powershell
cd taskpilot-platform-v3
python -m pytest -o addopts="" -q --junitxml=outputs/test-results-v3.xml
```

### 3.2 结果

```text
77 passed in 21.82s
```

- 原 64 个测试全部保留并通过；
- 新增 6 个回归测试：跨用户 key、列表隔离、并发 resume、completed 拒绝 resume、mark_succeeded 失败、旧表迁移；
- 新增 7 个多 Agent 对话测试：data→writing、research→writing、脚本化 research→data（证明无固定边）、max_turns 截断、工具越权拒绝、API 创建/读取/列表、多轮追问；
- 测试仍为离线 `MOCK_LLM=1`，不调用真实 API。

证据文件：`outputs/test-results-v3.xml`。

---

## 4. 真实 Qwen 接入

### 4.1 配置

```dotenv
QWEN_API_KEY=<from research-rag-agent/.env, not printed>
QWEN_BASE_URL=https://maas.qianwenaiapi.com/compatible-mode/v1
QWEN_CHAT_MODEL=qwen3.8-flash
MOCK_LLM=0
```

### 4.2 结构化 smoke

```powershell
$env:MOCK_LLM="0"
python scripts/test_llm.py
```

实际结果：

- 返回合法 `TaskPlan`；
- `task_type=data_analysis`；
- 熔断器 `CLOSED`，`total_successes=1`；
- 无重试、无 fallback。

证据：`outputs/real_llm_smoke.txt`。

### 4.3 端到端 smoke

```powershell
$env:MOCK_LLM="0"
python scripts/real_e2e_smoke.py
```

实际结果（最新一次）：

```text
HTTP 200
status=completed
task_type=data_analysis
trace=classify → context_retrieval → planner → react_decide → tool_execution
      → react_decide → writer → review → persist
tool_called=true, actual_tools=["analyze_csv"]
model_calls=4, tool_calls=1, tokens_used=7403
answer.metrics: MAE=1.35, RMSE=1.4089, R2=0.984876
```

证据：`outputs/real_e2e_smoke.json`、`outputs/real_e2e_smoke.txt`。

### 4.4 一个真实发现：同会话记忆会“替代”工具调用

同一 `user_id + session_id` 连续跑两次相同 CSV 任务时，第二次 `trace` 中没有 `tool_execution`：

```text
classify → context_retrieval → planner → react_decide → writer → review → persist
```

原因：第一次任务的短期记忆被 `ContextService` 召回，模型直接根据记忆里的指标作答，没有再调用 `analyze_csv`。

这不是“系统崩溃”，但它是**评测泄漏**：

- 如果评测复用同一 user/session，后一次任务会被前一次记忆污染；
- 真实评测必须每次使用 fresh user/session，或显式关闭短期记忆；
- `scripts/real_e2e_smoke.py` 已改为每次生成唯一 `user_id/session_id`，并强制断言 `tool_called`，否则退出码 3。

这个发现也说明：Memory 不是“越多越好”，评测设计必须隔离历史。

### 4.5 多 Agent 对话 smoke（v3.2）

```powershell
$env:MOCK_LLM="0"
python scripts/conversation_smoke.py
```

实际结果：

```text
HTTP 200
status=completed
agents=[data, writing]
handoffs: data → writing → finish
observations: analyze_csv ok
turn=2, model_calls=6, tokens=8605, 约 34s
final_answer: MAE=1.35, RMSE=1.4089, R2=0.984876，并说明样本量仅 8 条
```

Supervisor 第一次选择 data（缺少数值证据），第二次选择 writing（数值完成，转写成结论），第三次 finish。  
证据：`outputs/conversation_real_smoke.json`、`outputs/conversation_real_smoke.txt`。

---

## 5. 没有做/没有验证的

| 项 | 状态 |
|---|---|
| 任务队列 / Worker 租约 / 心跳 / reaper | 未实现（v3.3） |
| 运行中任务取消 | 未实现（v3.3） |
| 任务级单写者（多 worker） | 未实现；当前只有工具幂等 |
| 对话恢复/取消 | 未实现；`/conversations` 同步执行，未接 checkpoint |
| 真实模型质量评测（多任务/rubric/回归） | 未做；只有 3 类 smoke |
| Trace 节点耗时/`model_call_id`/任务级 Token 归因 | 未实现 |
| 多租户/RBAC | 未实现；`user_id` 仍可伪造 |
| Docker Compose / MCP / Gradio | 未在本环境验证 |
| 1 GiB 上传 / soak / 多主机共享 | 未做 |
| 真实模型 rate limit/401 行为 | 未专门注入；假客户端测试覆盖逻辑 |

---

## 6. 如何复现

```powershell
cd taskpilot-platform-v3
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 离线测试
python -m pytest -o addopts="" -q

# 真实模型（需自己的 key）
Copy-Item .env.example .env
# 编辑 .env：QWEN_API_KEY=...，MOCK_LLM=0
python scripts/test_llm.py
python scripts/real_e2e_smoke.py
python scripts/conversation_smoke.py

# 启动 API
uvicorn app.main:app --reload
```

---

## 7. 安全与 Key 处理

- `.env` 已在 `.gitignore` 中；
- 本报告、README、测试日志均不包含 Key；
- 交付压缩包**排除** `.env`；
- 真实 smoke 输出只包含模型名、Base URL、Token/延迟/指标，不含 Key；
- 生产环境应使用密钥服务/网关，不应把 Key 放在项目目录。

---

## 8. 结论

这次构建的目标不是“再造一个 Demo”，而是：

1. 让教程里的机制**真的能跑**；
2. 把审查发现的 4 个 P0 **先修掉**，不把已知缺陷带进新项目；
3. 用真实 Qwen 跑通 **planner → react → tool → writer**；
4. 新增 **Supervisor + 独立 Worker + 共享 transcript** 的自由对话多 Agent 路径，并跑通 **data → writing → finish**；
5. 明确哪些还没做（任务租约/取消/真实评测/多租户/对话恢复）。

当前状态：**可运行、可测试、可接真实模型、边界清楚**。  
下一步优先：对话恢复/取消 → 多 Agent 评测集 → 任务租约/取消 → Trace 节点耗时 → Docker 验证。

---

## 9. 相关文档

- 教程：`docs/v3/00_教程_v3.md`（上级工作区）
- 审查报告：`docs/v3/01_技术审查报告.md`
- P0/P1/P2：`docs/v3/03_问题清单_P0P1P2.md`
- 本项目的使用说明：`README.md`








