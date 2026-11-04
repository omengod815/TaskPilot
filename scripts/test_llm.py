"""One-call smoke test for the configured Qwen-compatible chat API."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.schemas.tasks import TaskPlan
from app.services.llm_service import AdaptiveLLMService


def main() -> None:
    settings = get_settings()
    if settings.mock_llm:
        print("当前 MOCK_LLM=1，不会调用真实 API。先验证工程主链路；真实模型测试前在 .env 中设置 MOCK_LLM=0。")
        return
    service = AdaptiveLLMService(settings=settings)
    messages = [
        {
            "role": "system",
            "content": "根据用户目标生成 TaskPlan 严格 JSON。只输出 JSON，不要 Markdown。Schema："
            + str(TaskPlan.model_json_schema()),
        },
        {"role": "user", "content": "目标：分析 CSV 数据中的预测误差。任务类型：data_analysis。"},
    ]
    result = service.generate_structured(messages, TaskPlan, user_id="smoke-test")
    print(result.model_dump_json(indent=2))
    print("调用指标：", service.metrics())


if __name__ == "__main__":
    main()
