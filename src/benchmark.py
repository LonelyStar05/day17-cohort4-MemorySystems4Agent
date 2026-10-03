from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    normalized_answer = answer.casefold()
    matched = sum(item.casefold() in normalized_answer for item in expected)
    if matched == 0:
        return 0.0
    if matched == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    if not answer.strip():
        return 0.0
    recall = recall_points(answer, expected)
    concise = 1.0 if len(answer) <= 600 else 0.5
    structured = 1.0 if "\n- " in f"\n{answer}" or len(expected) <= 1 else 0.5
    return round(0.75 * recall + 0.15 * concise + 0.10 * structured, 3)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
) -> BenchmarkRow:
    del config
    user_ids = {str(item["user_id"]) for item in conversations}
    initial_sizes = {
        user_id: agent.memory_file_size(user_id)
        if hasattr(agent, "memory_file_size")
        else 0
        for user_id in user_ids
    }
    thread_ids: set[str] = set()
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conversation in conversations:
        conversation_id = str(conversation["id"])
        user_id = str(conversation["user_id"])
        dialog_thread = f"{conversation_id}:dialog"
        thread_ids.add(dialog_thread)
        for turn in conversation.get("turns", []):
            agent.reply(user_id, dialog_thread, str(turn))

        for index, recall in enumerate(conversation.get("recall_questions", []), start=1):
            recall_thread = f"{conversation_id}:recall:{index}"
            thread_ids.add(recall_thread)
            result = agent.reply(user_id, recall_thread, str(recall["question"]))
            answer = str(result["response"])
            expected = [str(item) for item in recall.get("expected_contains", [])]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    final_sizes = {
        user_id: agent.memory_file_size(user_id)
        if hasattr(agent, "memory_file_size")
        else 0
        for user_id in user_ids
    }
    memory_growth = sum(
        max(0, final_sizes[user_id] - initial_sizes[user_id]) for user_id in user_ids
    )
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread_id) for thread_id in thread_ids),
        prompt_tokens_processed=sum(
            agent.prompt_token_usage(thread_id) for thread_id in thread_ids
        ),
        recall_score=(sum(recall_scores) / len(recall_scores) if recall_scores else 0.0),
        response_quality=(
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        ),
        memory_growth_bytes=memory_growth,
        compactions=sum(agent.compaction_count(thread_id) for thread_id in thread_ids),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    values = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.1%}",
            f"{row.response_quality:.1%}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(values, headers=headers, tablefmt="github")
    except ImportError:
        string_rows = [[str(value) for value in row] for row in values]
        widths = [
            max(len(headers[index]), *(len(row[index]) for row in string_rows))
            for index in range(len(headers))
        ]
        header = "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(headers)
        ) + " |"
        divider = "| " + " | ".join("-" * width for width in widths) + " |"
        body = [
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
            for row in string_rows
        ]
        return "\n".join([header, divider, *body])


def _run_suite(name: str, conversations: list[dict[str, Any]], config) -> list[BenchmarkRow]:
    with tempfile.TemporaryDirectory(prefix=f"memory-lab-{name}-") as temporary_dir:
        suite_config = replace(config, state_dir=Path(temporary_dir))
        baseline = BaselineAgent(suite_config, force_offline=True)
        advanced = AdvancedAgent(suite_config, force_offline=True)
        return [
            run_agent_benchmark("Baseline", baseline, conversations, suite_config),
            run_agent_benchmark("Advanced", advanced, conversations, suite_config),
        ]


def _analysis(rows: list[BenchmarkRow]) -> str:
    baseline, advanced = rows
    if baseline.prompt_tokens_processed:
        prompt_ratio = advanced.prompt_tokens_processed / baseline.prompt_tokens_processed
    else:
        prompt_ratio = 1.0
    if prompt_ratio <= 1:
        prompt_comparison = f"giảm {1 - prompt_ratio:.1%}"
    else:
        prompt_comparison = f"tăng {prompt_ratio - 1:.1%}"
    return (
        f"- Advanced recall: {advanced.recall_score:.1%}; Baseline recall: "
        f"{baseline.recall_score:.1%}. Persistent User.md tạo ra khác biệt qua thread mới.\n"
        f"- Advanced dùng {advanced.prompt_tokens_processed:,} prompt tokens so với "
        f"{baseline.prompt_tokens_processed:,} của Baseline "
        f"({prompt_comparison}).\n"
        f"- Advanced tạo file memory tăng {advanced.memory_growth_bytes} bytes và thực hiện "
        f"{advanced.compactions} lần compact: recall tốt hơn đổi lại bằng state và logic phức tạp hơn."
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_config(Path(__file__).resolve().parent.parent)
    standard = load_conversations(config.data_dir / "conversations.json")
    stress = load_conversations(config.data_dir / "advanced_long_context.json")

    standard_rows = _run_suite("standard", standard, config)
    stress_rows = _run_suite("stress", stress, config)

    print("# Standard Benchmark")
    print(format_rows(standard_rows))
    print("\n# Long-Context Stress Benchmark")
    print(format_rows(stress_rows))
    print("\n# Phân tích Standard")
    print(_analysis(standard_rows))
    print("\n# Phân tích Long Context")
    print(_analysis(stress_rows))
    print(
        "\n- Ở hội thoại ngắn, profile và summary có overhead cố định nên Advanced không "
        "nhất thiết rẻ hơn. Ở chuỗi dài, compact chặn tăng trưởng prompt kiểu toàn lịch sử.\n"
        "- User.md cần guardrail vì file có thể phình dần hoặc lưu nhầm fact; extractor hiện chỉ "
        "nhận mẫu có độ tin cậy cao, bỏ câu hỏi, và ghi đè location/profession khi có correction."
    )


if __name__ == "__main__":
    main()
