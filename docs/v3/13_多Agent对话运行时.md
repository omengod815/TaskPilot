# 自由对话式多 Agent 调度（v3.2）

> 目标：把“单图多角色编排”变成“Supervisor + 独立 Worker + 共享对话”的自由调度。
> 代码：`app/agents/conversation/`、`app/services/conversation_store.py`、`app/schemas/conversations.py`
> 入口：`POST /conversations`、`GET /conversations/{id}`、`GET /conversations`

---

## 1. 为什么要改

原来的 `/tasks` 是一条固定的 LangGraph：

```text
classify → context_retrieval → planner → react_decide ⇄ tool_execution
        → writer → review → persist
```

它适合“单任务、固定交付物”的场景，但本质仍是**一张图里的多个角色**：

- 下一个节点由边决定，不是由对话内容决定；
- 所有角色共享同一个 `AgentState`；
- Writer 永远是最后一个；
- 没有“谁该发言”的调度层。

“自由对话式多 Agent”要解决的是另一类问题：

> 用户只给一个目标；系统根据当前对话，决定“现在该谁说话、说什么、说完之后谁接”。

因此新增一条独立运行时，而不是把原图改得更复杂。

---

## 2. 运行时结构

```text
                  ┌─────────────────────────────┐
用户目标 ────────▶│  Shared Transcript (messages)│
                  └──────────────┬──────────────┘
                                 │
                        ┌────────▼────────┐
                        │   Supervisor    │
                        │ 决定下一位发言者 │
                        └───┬───┬───┬───┬─┘
                            │   │   │   │
              ┌─────────────┘   │   │   └─────────────┐
              ▼                 ▼   ▼                 ▼
        ┌──────────┐     ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Research │     │   Data   │ │ Writing  │ │ General  │
        │ search   │     │ analyze  │ │ 成稿      │ │ calculate│
        └────┬─────┘     └────┬─────┘ └────┬─────┘ └────┬─────┘
             └────────────────┴────────────┴────────────┘
                                 │
                        ┌────────▼────────┐
                        │  共享消息 / 观察 │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │ SQLite 会话存储  │
                        └─────────────────┘
```

关键点：

1. **没有固定的 Agent 边**。Supervisor 每一轮都可以选 research/data/writing/general/finish。
2. **Worker 彼此不直接调用**。它们读写同一份 transcript，避免“Agent 之间私聊失控”。
3. **Supervisor 不执行工具**。它只做调度，工具权限仍在 Worker/Registry。
4. **finish 不是直接输出**。由 Writing Agent 综合 transcript 生成最终答复，避免 Supervisor 编造。
5. **每轮持久化**。`conversations` 表保存 messages/handoffs/trace/budget。

---

## 3. 四个 Worker

| Agent | 职责 | 工具 | 不允许 |
|---|---|---|---|
| research | 检索、来源核验 | `search_web` | 编造来源 |
| data | CSV/数值分析 | `analyze_csv`、`safe_calculate` | 编造指标 |
| writing | 把证据转写成最终答复 | 无 | 新增事实 |
| general | 通用问答/简单计算 | `safe_calculate` | 编造事实 |

每个 Worker 有自己的 system prompt、工具白名单和 `task_type`。Registry 仍会做任务类型授权、参数校验、超时、熔断、幂等。

---

## 4. Supervisor 如何决定下一位

### 真实模型

Supervisor 调用 `SupervisorDecision`：

```python
class SupervisorDecision(BaseModel):
    next_agent: Literal["research", "data", "writing", "general", "finish"]
    instruction: str
    reason: str
```

输入包括：

- goal；
- 当前 turn / max_turns；
- 剩余预算；
- 已发言的 Agent；
- 最近 12 条 transcript。

规则：

- 需要外部资料 → research；
- 需要 CSV/指标 → data；
- 需要成稿 → writing；
- 简单通用问题 → general；
- 证据足够 → finish（交给 writing）。

### Mock / 测试

`MockSupervisor` 用关键词和已发言集合做确定性路由：

```text
无 Agent 发言：
  CSV/数据 → data
  调研/资料 → research
  写/报告 → writing
  其他 → general
已有证据且目标含“写/总结/报告”且 writing 未发言 → writing
否则 → finish
```

测试断言的是“路由 + 交接 + 最终答复”，不是真实模型质量。

---

## 5. 一次真实对话（2026-11-04）

请求：

```powershell
POST /conversations
{
  "goal": "分析 sample_predictions.csv 的 MAE、RMSE 和 R2，并写一段面向项目负责人的结论。",
  "user_id": "conv-smoke-...",
  "max_turns": 4
}
```

实际结果：

```text
status=completed
agents=[data, writing]
handoffs:
  0 → data     （缺少数值证据，先取指标）
  1 → writing  （数值完成，转写成结论）
  2 → finish
observations: analyze_csv ok
model_calls=6
tokens=8605
turn=2
```

最终答复包含 MAE=1.35、RMSE=1.4089、R2=0.984876，并主动说明“样本量仅 8 条，结论不足以代表长期/跨工况表现”。

证据：`outputs/conversation_real_smoke.json`、`outputs/conversation_real_smoke.txt`。

---

## 6. API

```powershell
# 创建并同步跑完一次对话
curl.exe -X POST http://127.0.0.1:8000/conversations `
  -H "Content-Type: application/json" `
  -d '{"goal":"调研 Agent 记忆机制并写一段总结","user_id":"u1","max_turns":6}'

# 读取
curl.exe "http://127.0.0.1:8000/conversations/<id>?user_id=u1"

# 列表
curl.exe "http://127.0.0.1:8000/conversations?user_id=u1"
```

响应包含：

- `messages`：user / supervisor / agent / tool 的共享 transcript；
- `handoffs`：每次“谁在什么理由下接手”；
- `observations`：工具结果；
- `trace`：supervisor/agent_turn 事件、耗时、调用数；
- `budget`：模型调用/工具/Token；
- `agents`：当前可调度 Agent 目录。

---

## 7. 与 `/tasks` 的关系

| 维度 | `/tasks` 单图 | `/conversations` 自由对话 |
|---|---|---|
| 控制流 | 固定条件边 | Supervisor 每轮动态选择 |
| 状态 | `AgentState` | 共享 transcript + handoffs |
| 角色 | 图节点 | 独立 WorkerAgent |
| 结束 | writer → review → persist | supervisor finish → writing |
| 持久化 | LangGraph Checkpoint + tasks | `conversations` 表 |
| 适用 | 单任务、固定交付 | 需要多角色协商/交接的任务 |

`/tasks` 没有删除；两条路径并存，避免破坏已有 70 个测试与单 Agent 稳定性。

---

## 8. 边界与已知问题

1. **不是“Agent 自治社会”**：Worker 不直接互发私信，所有消息经共享 transcript；Supervisor 是单点调度。
2. **Supervisor 可能误路由**：真实模型可能过早 finish 或重复选同一 Agent；当前只有 turn 上限和 prompt 约束，没有 handoff 评分。
3. **没有对话恢复/取消**：`/conversations` 是同步执行；进程崩溃后只能重新创建，未接 LangGraph checkpoint。
4. **没有每 Agent 私有记忆**：共享 transcript 是唯一上下文；长对话会膨胀。
5. **没有并行 Agent**：所有 Worker 串行发言；并行会引入共享状态合并问题。
6. **真实模型只跑了 1 次 smoke**：不能据此宣称多 Agent 比单 Agent 更好；需要固定评测集。
7. **`user_id` 仍可伪造**：和 `/tasks` 一样，认证仍是演示级。

---

## 9. 测试

```powershell
python -m pytest -o addopts="" -q
# 76 passed（2026-11-04）
```

新增覆盖：

- `test_data_agent_then_writer_finishes`：data → writing → finish；
- `test_research_then_writer_hands_off`：research → writing；
- `test_max_turns_stops_conversation`：max_turns 截断为 partial；
- `test_worker_rejects_tool_outside_its_profile`：工具越权拒绝；
- `test_conversation_api_create_get_list`：API 创建/读取/列表/跨用户 404；\n- `test_runtime_follows_supervisor_choice_not_fixed_order`：脚本化 Supervisor 强制 research→data→finish，证明运行时没有固定边。

---

## 10. 下一步

1. 对话恢复：把 `ConversationState` 每轮写 checkpoint，支持 `POST /conversations/{id}/resume`。
2. 取消：协作式 cancel token。
3. Supervisor 评测：固定 20 条多 Agent 任务，统计 handoff 正确率、完成率、Token。
4. 每 Agent 私有摘要：共享 transcript 只保留最近 N 条 + 各 Agent 摘要，控制上下文。
5. 并行 Worker：只允许只读 Agent 并行，写操作仍串行。
