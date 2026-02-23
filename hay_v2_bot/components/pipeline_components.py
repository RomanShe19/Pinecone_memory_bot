from __future__ import annotations

from datetime import datetime
import json
from typing import Any, Dict, List, Optional

from haystack import Document, component
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage


@component
class ConversationDocumentBuilder:
    """Builds a Haystack document from message-response pair."""

    @component.output_types(documents=List[Document])
    def run(
        self,
        user_id: int,
        username: str,
        user_message: str,
        assistant_response: str,
    ) -> Dict[str, Any]:
        timestamp = datetime.now().isoformat()
        conversation_text = (
            f"Пользователь ({username}): {user_message}\nАссистент: {assistant_response}"
        )
        doc = Document(
            content=conversation_text,
            meta={
                "source_type": "conversation",
                "user_id": str(user_id),
                "username": username,
                "timestamp": timestamp,
                "user_message": user_message,
                "assistant_response": assistant_response,
            },
        )
        return {"documents": [doc]}


@component
class FileMetadataEnricher:
    """Adds file- and user-level metadata to Docling chunks."""

    @component.output_types(documents=List[Document])
    def run(
        self,
        documents: List[Document],
        user_id: int,
        username: str,
        file_name: str,
    ) -> Dict[str, Any]:
        enriched: List[Document] = []
        for idx, doc in enumerate(documents):
            chunk_meta = dict(doc.meta or {})
            chunk_meta.update(
                {
                    "source_type": "file",
                    "user_id": str(user_id),
                    "username": username,
                    "file_name": file_name,
                    "chunk_index": idx,
                    "page_number": self._extract_page_number(doc.meta),
                }
            )
            doc.meta = self._sanitize_meta(chunk_meta)
            enriched.append(doc)
        return {"documents": enriched}

    @staticmethod
    def _extract_page_number(meta: Optional[Dict[str, Any]]) -> Optional[int]:
        if not meta:
            return None
        dl_meta = meta.get("dl_meta")
        if isinstance(dl_meta, dict):
            try:
                doc_items = dl_meta.get("meta", {}).get("doc_items", [])
                first_item = doc_items[0] if doc_items else None
                prov = first_item.get("prov", []) if isinstance(first_item, dict) else []
                first_prov = prov[0] if prov else None
                page = first_prov.get("page_no") if isinstance(first_prov, dict) else None
                if isinstance(page, int):
                    return page
            except Exception:
                return None
        return None

    @staticmethod
    def _sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
        # Pinecone integration supports only str/int/bool/list[str] metadata values.
        # Normalize and drop unsupported fields to avoid noisy writer warnings.
        clean_meta: Dict[str, Any] = {}
        for key, value in meta.items():
            if value is None:
                continue
            if key.startswith("_"):
                continue
            if isinstance(value, (str, int, bool)):
                clean_meta[key] = value
                continue
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                clean_meta[key] = value
                continue
            if key == "dl_meta":
                # Keep docling grounding as serialized JSON for compatibility with Pinecone meta types.
                try:
                    clean_meta["dl_meta_json"] = json.dumps(value, ensure_ascii=False)[:4000]
                except Exception:
                    clean_meta["dl_meta_json"] = str(value)[:4000]
        return clean_meta


@component
class RetrievedContextFormatter:
    """Formats retrieved documents into a user prompt."""

    def __init__(self, max_docs: int = 5, max_chars: int = 4000):
        self.max_docs = max_docs
        self.max_chars = max_chars

    @component.output_types(prompt=str)
    def run(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        selected_docs = documents[: self.max_docs]
        context_blocks = []
        for doc in selected_docs:
            source_type = (doc.meta or {}).get("source_type", "unknown")
            file_name = (doc.meta or {}).get("file_name")
            page_number = (doc.meta or {}).get("page_number")
            source_hint = f"type={source_type}"
            if file_name:
                source_hint += f", file={file_name}"
            if page_number:
                source_hint += f", page={page_number}"
            context_blocks.append(f"[{source_hint}] {doc.content}")

        context_text = "\n\n".join(context_blocks)
        if len(context_text) > self.max_chars:
            context_text = context_text[: self.max_chars] + "..."

        prompt = (
            f"{query}\n\n"
            "Контекст из истории и документов пользователя:\n"
            f"{context_text if context_text else 'Контекст не найден.'}"
        )
        return {"prompt": prompt}


@component
class AgentResponder:
    """Runs Haystack Agent and extracts text + optional image_url."""

    def __init__(self, agent: Agent):
        self.agent = agent

    @component.output_types(response_text=str, image_url=Optional[str])
    def run(self, prompt: str) -> Dict[str, Any]:
        result = self.agent.run(messages=[ChatMessage.from_user(prompt)])
        messages = result.get("messages", [])
        response_text = messages[-1].text if messages else ""
        image_url = None

        for message in messages:
            message_meta = getattr(message, "meta", None) or {}
            tool_output = message_meta.get("tool_output", {})
            maybe_image_url = tool_output.get("image_url")
            if maybe_image_url:
                image_url = maybe_image_url
                break

        return {"response_text": response_text, "image_url": image_url}

