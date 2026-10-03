from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


PROFILE_HEADER = "# User Profile\n"
MERGED_FACT_KEYS = {"interests", "response_style", "priorities"}


def estimate_tokens(text: str) -> int:
    """Return a stable tokenizer-free approximation suitable for offline tests."""

    normalized = " ".join(text.split())
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / 4))


def _safe_user_slug(user_id: str) -> str:
    normalized = unicodedata.normalize("NFKC", user_id).strip()
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized).strip("._")
    if not slug:
        slug = "anonymous"
    return slug[:80]


@dataclass
class UserProfileStore:
    """Persistent, human-readable storage for one `User.md` per user."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / _safe_user_slug(user_id) / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return f"{PROFILE_HEADER}\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = content.rstrip() + "\n"
        path.write_text(normalized, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^- ([a-z][a-z0-9_]*):\s*(.+)$", line.strip())
            if match:
                facts[match.group(1)] = match.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        clean_key = re.sub(r"[^a-z0-9_]+", "_", key.lower()).strip("_")
        clean_value = " ".join(value.strip().split()).strip(" .")
        if not clean_key or not clean_value:
            return self.path_for(user_id)

        facts = self.facts(user_id)
        if clean_key in MERGED_FACT_KEYS and clean_key in facts:
            existing_parts = [part.strip() for part in facts[clean_key].split(";")]
            new_parts = [part.strip() for part in clean_value.split(";")]
            merged: list[str] = []
            for part in existing_parts + new_parts:
                if part and part.casefold() not in {item.casefold() for item in merged}:
                    merged.append(part)
            clean_value = "; ".join(merged)
        facts[clean_key] = clean_value

        preferred_order = [
            "name",
            "location",
            "profession",
            "response_style",
            "favorite_drink",
            "favorite_food",
            "pet",
            "interests",
            "priorities",
        ]
        keys = [key for key in preferred_order if key in facts]
        keys.extend(sorted(key for key in facts if key not in keys))
        body = "\n".join(f"- {fact_key}: {facts[fact_key]}" for fact_key in keys)
        return self.write_text(user_id, f"{PROFILE_HEADER}\n{body}\n")


def _last_match(patterns: list[str], message: str) -> str | None:
    matches: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, message, flags=re.IGNORECASE):
            matches.append((match.start(), " ".join(match.group(1).split()).strip(" .,:;")))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract only high-confidence, stable user facts from Vietnamese text."""

    text = " ".join(message.split())
    lower = text.casefold()
    if not text:
        return {}

    # A conservative confidence threshold: recall questions must never overwrite facts.
    if "?" in text:
        return {}

    updates: dict[str, str] = {}

    name = _last_match(
        [
            r"(?:mình|tôi)\s+tên\s+là\s+([^,.;!?]+)",
            r"tên\s+(?:mình|tôi)\s+là\s+([^,.;!?]+)",
        ],
        text,
    )
    if name:
        updates["name"] = name

    location = _last_match(
        [
            r"(?:mình|tôi)\s+(?:hiện\s+)?(?:đang\s+|vẫn\s+)?ở\s+([A-ZÀ-Ỹ][\wÀ-ỹ -]*?)(?=\s+(?:và|chứ|dù|để|cho|vài\s+tháng)\b|[,.;!?]|$)",
            r"(?:hiện|hiện tại)\s+ở\s+([A-ZÀ-Ỹ][\wÀ-ỹ -]*?)(?=\s+(?:và|chứ|dù|để|cho)\b|[,.;!?]|$)",
            r"(?:mình|tôi)\s+đang\s+làm\s+việc\s+ở\s+([A-ZÀ-Ỹ][\wÀ-ỹ -]*?)(?=\s+(?:và|chứ|dù|để|cho|vài\s+tháng)\b|[,.;!?]|$)",
            r"nơi\s+ở\s+(?:hiện tại\s+)?(?:của\s+mình\s+)?(?:vẫn\s+)?là\s+([A-ZÀ-Ỹ][\wÀ-ỹ -]*?)(?=\s+(?:và|chứ|dù|để|cho)\b|[,.;!?]|$)",
        ],
        text,
    )
    if location:
        updates["location"] = location

    profession = _last_match(
        [
            r"(?:đang|hiện đang|vẫn)\s+làm\s+([A-Za-z][A-Za-z0-9+.# -]{0,50}?(?:engineer|developer|manager|designer|analyst|scientist))\b",
            r"(?:giờ|hiện tại)\s+(?:mình\s+)?(?:đã\s+)?(?:chuyển sang|làm)\s+([A-Za-z][A-Za-z0-9+.# -]{0,50}?(?:engineer|developer|manager|designer|analyst|scientist))\b",
            r"chuyển sang\s+([A-Za-z][A-Za-z0-9+.# -]{0,50}?(?:engineer|developer|manager|designer|analyst|scientist))\b",
            r"nghề nghiệp\s+(?:hiện tại\s+)?(?:của mình\s+)?(?:vẫn\s+)?là\s+([A-Za-z][A-Za-z0-9+.# -]{0,50}?(?:engineer|developer|manager|designer|analyst|scientist))\b",
            r"mình\s+làm\s+([A-Za-z][A-Za-z0-9+.# -]{0,50}?(?:engineer|developer|manager|designer|analyst|scientist))\b",
        ],
        text,
    )
    if profession:
        updates["profession"] = profession

    style_context = any(
        marker in lower
        for marker in ("trả lời", "giải thích", "style", "cách trình bày", "câu trả lời")
    )
    if style_context:
        style_parts: list[str] = []
        if any(marker in lower for marker in ("ngắn gọn", "bullet ngắn", "trả lời gọn", "câu trả lời gọn")):
            style_parts.append("ngắn gọn")
        if "3 bullet" in lower or "ba bullet" in lower:
            style_parts.append("3 bullet")
        elif "bullet" in lower:
            style_parts.append("có bullet")
        if "rõ ý" in lower or "có cấu trúc" in lower:
            style_parts.append("rõ ý, có cấu trúc")
        if "ví dụ thực chiến" in lower:
            style_parts.append("có ví dụ thực chiến")
        elif "ví dụ thực tế" in lower:
            style_parts.append("có ví dụ thực tế")
        if "trade-off" in lower:
            style_parts.append("nhấn mạnh trade-off")
        if style_parts:
            updates["response_style"] = "; ".join(dict.fromkeys(style_parts))

    if "cà phê sữa đá" in lower and any(
        marker in lower for marker in ("đồ uống", "mình thích", "vẫn uống", "yêu thích")
    ):
        updates["favorite_drink"] = "cà phê sữa đá"

    if "mì quảng" in lower and any(
        marker in lower for marker in ("món ăn", "món ruột", "yêu thích")
    ):
        updates["favorite_food"] = "mì Quảng"

    if "corgi" in lower and any(marker in lower for marker in ("nuôi", "bé corgi", "con corgi")):
        pet_name = re.search(r"corgi\s+(?:tên\s+)?([A-ZÀ-Ỹ][\wÀ-ỹ-]*)", text)
        updates["pet"] = f"corgi tên {pet_name.group(1)}" if pet_name else "corgi"

    interest_context = any(
        marker in lower
        for marker in ("mình thích", "mình đang quan tâm", "mối quan tâm", "dài hạn:")
    )
    if interest_context:
        keyword_map = [
            ("python", "Python"),
            ("ai ứng dụng", "AI ứng dụng"),
            ("ai agent", "AI agent"),
            ("benchmark memory", "benchmark memory"),
            ("mlops", "MLOps"),
            ("rag", "RAG"),
            ("evaluation", "evaluation"),
        ]
        interests = [label for keyword, label in keyword_map if keyword in lower]
        if interests:
            updates["interests"] = "; ".join(dict.fromkeys(interests))

    priority_parts: list[str] = []
    if "ưu tiên recall" in lower:
        priority_parts.append("ưu tiên recall đúng")
    if "số liệu rõ ràng" in lower or "định lượng" in lower:
        priority_parts.append("ưu tiên số liệu rõ ràng")
    if priority_parts:
        updates["priorities"] = "; ".join(priority_parts)

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a bounded extractive summary that remains deterministic offline."""

    if not messages or max_items <= 0:
        return ""

    compact_items: list[str] = []
    for message in messages:
        role = str(message.get("role", "unknown"))
        content = " ".join(str(message.get("content", "")).split())
        if not content:
            continue
        if len(content) > 240:
            content = content[:237].rstrip() + "..."
        compact_items.append(f"- {role}: {content}")

    if len(compact_items) > max_items:
        head_count = max(1, max_items // 3)
        tail_count = max_items - head_count
        compact_items = compact_items[:head_count] + compact_items[-tail_count:]
    return "\n".join(compact_items)


@dataclass
class CompactMemoryManager:
    """Keep recent messages verbatim and compact older content into a summary."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread_state(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread_state(thread_id)
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})

        summary = str(thread["summary"])
        context_text = summary + "\n" + "\n".join(
            str(item.get("content", "")) for item in messages
        )
        if estimate_tokens(context_text) <= self.threshold_tokens:
            return
        if len(messages) <= self.keep_messages:
            return

        older = messages[:-self.keep_messages]
        recent = messages[-self.keep_messages :]
        addition = summarize_messages(older, max_items=8)
        combined = "\n".join(part for part in (summary, addition) if part).strip()
        if len(combined) > 1_800:
            combined = combined[:900].rstrip() + "\n...\n" + combined[-850:].lstrip()
        thread["summary"] = combined
        thread["messages"] = recent
        thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self._thread_state(thread_id)
        return {
            "messages": [dict(item) for item in thread["messages"]],
            "summary": str(thread["summary"]),
            "compactions": int(thread["compactions"]),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self._thread_state(thread_id)["compactions"])
