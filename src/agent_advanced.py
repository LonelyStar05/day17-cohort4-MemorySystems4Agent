from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model, normalize_provider


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: persistent profile plus compact per-thread memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _persist_updates(self, user_id: str, message: str) -> dict[str, str]:
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)
        return updates

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        updates = self._persist_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        answer = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", answer)
        generated_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + generated_tokens
        return {
            "response": answer,
            "token_usage": generated_tokens,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_processed": self.thread_prompt_tokens[thread_id],
            "memory_path": str(self.profile_store.path_for(user_id)),
            "profile_updates": updates,
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        message_text = "\n".join(
            str(item.get("content", "")) for item in context["messages"]
        )
        combined = "\n".join(
            (
                self.profile_store.read_text(user_id),
                str(context["summary"]),
                message_text,
            )
        )
        return estimate_tokens(combined)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        del thread_id
        facts = self.profile_store.facts(user_id)
        lower = message.casefold()
        requested: list[str] = []
        checks = [
            ("name", ("tên", "dũngct")),
            ("profession", ("nghề", "làm nghề", "backend", "mlops", "product manager")),
            ("location", ("ở đâu", "nơi ở", "huế", "hà nội")),
            ("response_style", ("style", "kiểu trả lời", "trả lời mình thích")),
            ("favorite_drink", ("đồ uống",)),
            ("favorite_food", ("món ăn",)),
            ("pet", ("nuôi con gì", "corgi", "con gì")),
            ("interests", ("mối quan tâm", "quan tâm kỹ thuật", "hai mối", "tóm tắt")),
        ]
        for key, markers in checks:
            if any(marker in lower for marker in markers):
                requested.append(key)

        available = [(key, facts[key]) for key in requested if key in facts]
        if available:
            labels = {
                "name": "Tên",
                "profession": "Nghề nghiệp hiện tại",
                "location": "Nơi ở hiện tại",
                "response_style": "Style trả lời",
                "favorite_drink": "Đồ uống yêu thích",
                "favorite_food": "Món ăn yêu thích",
                "pet": "Thú cưng",
                "interests": "Mối quan tâm",
            }
            return "\n".join(f"- {labels[key]}: {value}" for key, value in available)
        if requested:
            return "Mình chưa có fact ổn định phù hợp trong User.md."

        updates = extract_profile_updates(message)
        if updates:
            labels = ", ".join(updates)
            return f"Đã cập nhật các fact ổn định trong User.md: {labels}."
        return "Mình đã ghi nhận ngữ cảnh gần đây; chỉ fact ổn định mới được lưu vào User.md."

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        updates = self._persist_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        context = self.compact_memory.context(thread_id)
        system_prompt = (
            "Bạn là trợ lý có memory. Dùng profile và summary bên dưới, ưu tiên fact mới nhất.\n\n"
            f"PROFILE:\n{self.profile_store.read_text(user_id)}\n"
            f"SUMMARY:\n{context['summary']}"
        )
        messages = [{"role": "system", "content": system_prompt}] + list(context["messages"])
        result = self.langchain_agent.invoke(messages)
        answer = _message_text(result)
        self.compact_memory.append(thread_id, "assistant", answer)
        generated_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + generated_tokens
        return {
            "response": answer,
            "token_usage": generated_tokens,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_processed": self.thread_prompt_tokens[thread_id],
            "memory_path": str(self.profile_store.path_for(user_id)),
            "profile_updates": updates,
            "compactions": self.compaction_count(thread_id),
        }

    def _maybe_build_langchain_agent(self):
        if self.force_offline:
            return None
        provider = normalize_provider(self.config.model.provider)
        if provider != "ollama" and not self.config.model.api_key:
            return None
        try:
            return build_chat_model(self.config.model)
        except (RuntimeError, ValueError):
            return None


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)
