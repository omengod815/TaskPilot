# 对话恢复 / 取消与固定多 Agent 评测（v3.4）

> 这是 v3.2/v3.3 自由对话式多 Agent 的可靠性补充：真实模型多轮 smoke、协作式恢复/取消、固定回归评测集。
> 代码：`app/services/conversation_store.py`、`app/agents/conversation/runtime.py`、`scripts/conversation_multiturn_smoke.py`、`scripts/eval_conversations.py`、`evals/conversation_suite.json`
> API：`POST /conversations/{id}/messages`、`POST /conversations/{id}/resume`、`POST /conversations/{id}/cancel`

---

## 1. 真实模型多轮 smoke

脚本：`scripts/conversation_multiturn_smoke.py`

流程：

```text
创建对话：分析 sample_predictions.csv 的 MAE/RMSE/R2
        → data 调 analyze_csv
        → writing 生成第一轮答复
追问：请用一句话总结上面的指标，并说明样本量局限
        → Supervisor 基于新消息重新调度
        → writing 生成第二轮答复
```

2026-11-24 实际结果（Qwen `qwen3.8-flash`）：

```text
first_status=completed
follow_status=completed
user_messages=2
agents=[data, writing, writing]
turn=3, max_turns=4
model_calls=8
tokens=11441
first_answer  = 包含 MAE=1.35、RMSE=1.4089、R2=0.984876，并说明 8 行样本局限
follow_answer = “MAE 为 1.35、RMSE 为 1.4089、R2 为 0.984876，但样本量仅 8 行……”
```

证据：

- `outputs/conversation_multiturn_real_smoke.json`
- `outputs/conversation_multiturn_real_smoke.txt`
- `outputs/conversation_multiturn_console.log`

注意：这是一次 smoke，不是质量评测；它证明“同一对话的追问会重新调度”，不证明多轮对话一定优于单轮。

---

## 2. 对话恢复

### 2.1 语义

- 只有 `partial` / `failed` / `interrupted` / `cancelled` 可恢复。
- `completed`、`running` 拒绝恢复；`running` 返回 409，避免两个请求同时跑。
- 恢复时原子领取：`ConversationStore.claim_for_resume()` 用 `BEGIN IMMEDIATE` 读状态、校验、写回 `running`。
- 保留原 transcript、observations、handoffs、budget。
- `extra_turns` 决定本次追加的 worker 轮数：
  `max_turns = max(old_max_turns, current_turn + extra_turns)`。
- 恢复不是“从某个 LangGraph 节点重放”，而是从已保存的对话状态继续下一轮 Supervisor 决策。

### 2.2 API

```powershell
curl.exe -X POST http://127.0.0.1:8000/conversations/<id>/resume `
  -H "Content-Type: application/json" `
  -d '{"user_id":"u1","extra_turns":2}'
```

### 2.3 边界

- 没有节点级 checkpoint；如果进程在 worker 执行中崩溃，最多丢失当前未持久化的那一轮。
- 恢复后 Supervisor 可能再次误路由；当前有 turn 上限和重复 Agent 保护，但没有 handoff 评分。

---

## 3. 协作式取消

### 3.1 语义

- `POST /conversations/{id}/cancel` 只对 `running` 对话生效。
- 取消写入 `cancel_requested=true`；运行时在**每个 turn 边界**检查 DB 标记，命中后：
  - `status=cancelled`
  - `stop_reason=cancelled_by_user`
  - `final_answer` 使用当前已有 Agent 答复的 best effort
  - 不再调用新的 Supervisor/Worker。
- 已经结束的对话返回 409。
- 不能强杀正在进行的模型/工具调用；取消是协作式的，最坏要等当前调用返回。

### 3.2 API

```powershell
curl.exe -X POST "http://127.0.0.1:8000/conversations/<id>/cancel?user_id=u1"
```

### 3.3 状态机

```text
running ── cancel_requested ──▶ cancelling ── turn boundary ──▶ cancelled
   │                                                          
   ├── finish ──▶ completed
   ├── max_turns/budget ──▶ partial
   └── exception ──▶ failed
```

`cancelled` 可以恢复；`completed` 不能取消，也不能恢复。

---

## 4. 固定多 Agent 评测集

### 4.1 套件

`evals/conversation_suite.json` 固定 8 条任务，覆盖：

| id | 目标 | 期望 Agent | 必需工具 |
|---|---|---|---|
| csv_metrics | CSV 指标 | data, writing | analyze_csv |
| research_summary | 调研并总结 | research, writing | search_web |
| safe_calc | 安全计算 | general, writing | safe_calculate |
| writing_only | 只成稿 | writing | 无 |
| data_limitations | 指标 + 样本局限 | data, writing | analyze_csv |
| mixed_research_data | 调研 + CSV + 成稿 | research, data, writing | search_web, analyze_csv |
| general_explain | 通用解释 | general, writing | 无 |
| csv_follow_up | CSV + 追问 | data, writing | analyze_csv |

判定字段：

- `expected_agents`：必须出现的 Agent（不要求顺序）；
- `required_tools`：必须成功执行的工具；
- `answer_contains` / `answer_any`：最终答复必须包含的关键词；
- `max_model_calls`：预算合理性检查；
- `follow_up`：可选的第二轮用户消息。

### 4.2 运行

```powershell
# 确定性模式（CI/离线）
python scripts/eval_conversations.py

# 真实模型（消耗 Qwen 配额）
python scripts/eval_conversations.py --real

# 单任务调试
python scripts/eval_conversations.py --real --task research_summary
```

输出：

- `outputs/conversation_eval.json`
- `outputs/conversation_eval.md`
- mock 证据另存：`outputs/conversation_eval_mock.json` / `.md`

### 4.3 结果

2026-11-24，Qwen `qwen3.8-flash`：

| 模式 | 通过 | 完成率 | Agent 命中 | 工具命中 | 答案命中 | 平均模型调用 | 平均 Token | 平均耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mock | 8/8 | 100% | 100% | 100% | 100% | 0 | 0 | 0.08s |
| real | 8/8 | 100% | 100% | 100% | 100% | 6.5 | 11413.1 | 59.93s |

真实模式 wall time 约 480s（8 条任务，串行）。

> 这是固定回归集，不是统计基准。8 条任务、单模型、单次运行，不能据此宣称“多 Agent 普遍优于单 Agent”。

### 4.4 真实评测暴露的问题与修复

第一次真实运行只有 4/8：

- Supervisor 重复选择同一 Agent（如 research→research、writing→writing），耗尽预算或 turn；
- 部分任务 `max_turns` 太小，`writing` 完成后没有机会 `finish`；
- `max_model_calls` 估计过低，实际 5–9 次调用被误判为失败。

修复：

1. `LLMSupervisor` 增加**重复 Agent 保护**：如果模型选择的 Agent 本轮已经发过言，降级到确定性策略；确定性策略仍重复则直接 `finish`。记录 `degraded_reasons`。
2. 评测套件按真实任务复杂度调整 `max_turns` / `max_model_calls`（不是放宽答案标准）。
3. `MockSupervisor` 支持“research→data→writing”混合路由，供离线回归。
4. `writing` 的 mock 答复补充 `safe_calculate` 观察，避免计算结果在最终答复中丢失。

---

## 5. 测试

```powershell
python -m pytest -o addopts="" -q
# 83 passed（2026-11-24）
```

新增覆盖：

- `test_partial_conversation_can_resume`
- `test_cancel_flag_stops_runtime_before_next_turn`
- `test_cancel_endpoint_marks_running_conversation`
- `test_resume_api_and_completed_rejection`
- `test_supervisor_repeat_guard_does_not_call_same_agent`
- `test_conversation_eval_suite_is_well_formed`

---

## 6. 已知限制

1. **取消是协作式的**：不能中断已经发出的模型/工具调用。
2. **恢复不是节点级重放**：没有 LangGraph checkpoint 的逐节点恢复。
3. **评测只有 8 条**：覆盖主路径，不足以证明统计显著性。
4. **supervisor 重复保护是兜底**：不是真正的 handoff 质量评估。
5. **`user_id` 仍可伪造**：认证仍是演示级。
6. **真实评测串行、单机**：没有并发/多租户/成本控制。

---

## 7. 下一步

1. 把 `cancel_requested` 扩展为可中断的 HTTP/模型调用（需要异步执行器）。
2. 对话状态写 LangGraph checkpoint，支持节点级恢复。
3. 评测集扩到 20–30 条，加入期望工具调用顺序、handoff 质量和成本上限。
4. 每 Agent 私有摘要，控制长对话上下文。
5. 真实模型评测纳入 CI（手动触发，避免每天消耗配额）。

