from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model, normalize_provider


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: full short-term thread history without persistent memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        del user_id  # Baseline deliberately has no cross-thread user store.
        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = estimate_tokens(
            "\n".join(item["content"] for item in session.messages)
        )
        session.prompt_tokens_processed += prompt_tokens

        answer = self._offline_response(session, message)
        session.messages.append({"role": "assistant", "content": answer})
        generated_tokens = estimate_tokens(answer)
        session.token_usage += generated_tokens
        return {
            "response": answer,
            "token_usage": generated_tokens,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_processed": session.prompt_tokens_processed,
        }

    def _offline_response(self, session: SessionState, message: str) -> str:
        facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))

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
            ("interests", ("mối quan tâm", "quan tâm kỹ thuật", "tóm tắt")),
        ]
        for key, markers in checks:
            if any(marker in lower for marker in markers):
                requested.append(key)

        available = [(key, facts[key]) for key in requested if key in facts]
        if requested and not available:
            return "Mình chưa có thông tin đó trong thread hiện tại."
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
        return "Mình đã ghi nhận nội dung này trong thread hiện tại."

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = estimate_tokens(
            "\n".join(item["content"] for item in session.messages)
        )
        session.prompt_tokens_processed += prompt_tokens
        result = self.langchain_agent.invoke(session.messages)
        answer = _message_text(result)
        session.messages.append({"role": "assistant", "content": answer})
        generated_tokens = estimate_tokens(answer)
        session.token_usage += generated_tokens
        return {
            "response": answer,
            "token_usage": generated_tokens,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_processed": session.prompt_tokens_processed,
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
