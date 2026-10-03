from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore, extract_profile_updates
from model_provider import normalize_provider


def make_config(tmp_path: Path):
    config = load_config(Path(__file__).resolve().parent.parent)
    return replace(
        config,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=80,
        compact_keep_messages=2,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.read_text("user/../01").startswith("# User Profile")

    path = store.write_text("user/../01", "# User Profile\n\n- name: Lan")
    assert path.name == "User.md"
    assert path.is_file()
    assert store.edit_text("user/../01", "Lan", "Linh") is True
    assert "Linh" in store.read_text("user/../01")
    assert store.edit_text("user/../01", "missing", "value") is False

    store.upsert_fact("user/../01", "location", "Huế")
    store.upsert_fact("user/../01", "location", "Đà Nẵng")
    assert store.facts("user/../01")["location"] == "Đà Nẵng"


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    long_turn = "Mình đang phân tích trade-off memory và token cost. " * 12
    for _ in range(6):
        agent.reply("compact-user", "long-thread", long_turn)

    context = agent.compact_memory.context("long-thread")
    assert agent.compaction_count("long-thread") > 0
    assert context["summary"]
    assert len(context["messages"]) <= agent.config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    fact = "Mình tên là Lan và đang làm MLOps engineer."

    baseline.reply("lan", "session-a", fact)
    advanced.reply("lan", "session-a", fact)
    baseline_answer = baseline.reply("lan", "session-b", "Mình tên gì và làm nghề gì?")[
        "response"
    ]
    advanced_answer = advanced.reply(
        "lan", "session-b", "Mình tên gì và làm nghề gì?"
    )["response"]

    assert "Lan" not in baseline_answer
    assert "MLOps engineer" not in baseline_answer
    assert "Lan" in advanced_answer
    assert "MLOps engineer" in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    long_turn = (
        "Đây là một lượt hội thoại dài mô tả readiness, externality, uncertainty và "
        "efficiency để benchmark prompt context. " * 10
    )
    for index in range(14):
        message = f"Lượt {index}: {long_turn}"
        baseline.reply("stress", "stress-thread", message)
        advanced.reply("stress", "stress-thread", message)

    assert advanced.compaction_count("stress-thread") > 0
    assert advanced.prompt_token_usage("stress-thread") < baseline.prompt_token_usage(
        "stress-thread"
    )


def test_conflict_handling_and_noise_filter(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply(
        "dungct",
        "one",
        "Mình ở Huế và đang làm backend engineer cho startup AI.",
    )
    agent.reply(
        "dungct",
        "two",
        "Giờ mình đang ở Đà Nẵng và chuyển sang MLOps engineer.",
    )
    agent.reply(
        "dungct",
        "three",
        "Product manager chỉ là câu đùa; Hà Nội là nơi đi họp, không phải nơi ở hiện tại.",
    )
    answer = agent.reply(
        "dungct",
        "fresh",
        "Nghề nghiệp và nơi ở hiện tại của mình là gì?",
    )["response"]

    assert "MLOps engineer" in answer
    assert "Đà Nẵng" in answer
    assert "backend engineer" not in answer
    assert "product manager" not in answer.casefold()
    assert "Hà Nội" not in answer


def test_questions_do_not_become_profile_facts() -> None:
    assert extract_profile_updates("Mình tên gì và đang ở đâu?") == {}
    assert extract_profile_updates("Bạn có biết DũngCT không?") == {}


def test_provider_aliases() -> None:
    assert normalize_provider("anthorpic") == "anthropic"
    assert normalize_provider("Open_Router") == "openrouter"
