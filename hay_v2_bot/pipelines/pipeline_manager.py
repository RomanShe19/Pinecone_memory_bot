from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from haystack import Document, Pipeline
from haystack.components.agents import Agent
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.tools import ComponentTool
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from hay_v2_bot.components.external_tools import DogFactFetcher, DogImageFetcher, WeatherFetcher
from hay_v2_bot.components.pipeline_components import (
    AgentResponder,
    ConversationDocumentBuilder,
    FileMetadataEnricher,
    RetrievedContextFormatter,
)
from hay_v2_bot.config import Settings

try:
    from docling_haystack.converter import DoclingConverter, ExportType

    DOCLING_AVAILABLE = True
except ImportError:
    DoclingConverter = None  # type: ignore[assignment]
    ExportType = None  # type: ignore[assignment]
    DOCLING_AVAILABLE = False


class PipelineManager:
    """Creates and executes all bot pipelines."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._configure_hf_windows_symlink_fallback()
        self.document_store = PineconeDocumentStore(
            index=settings.pinecone_index_name,
            metric="cosine",
            dimension=1536,
            spec={
                "serverless": {
                    "region": settings.pinecone_region,
                    "cloud": settings.pinecone_cloud,
                }
            },
        )

        self.agent = self._build_agent()
        self.summary_generator = OpenAIChatGenerator(
            model=settings.chat_model,
            api_key=Secret.from_token(settings.openai_api_key),
            api_base_url=settings.openai_base_url,
        )

        self.generation_pipeline = self._build_generation_pipeline()
        self.conversation_pipeline = self._build_conversation_pipeline()
        self._file_ingestion_available = False
        self._file_ingestion_reason = (
            "Компонент обработки файлов недоступен: установите зависимости "
            "'docling' и 'docling-haystack'."
        )
        self.ingestion_pipeline = Pipeline()
        if DOCLING_AVAILABLE:
            try:
                self.ingestion_pipeline = self._build_ingestion_pipeline()
                self._file_ingestion_available = True
                self._file_ingestion_reason = ""
            except Exception as exc:
                self._file_ingestion_reason = (
                    "Компонент обработки файлов временно недоступен: "
                    f"{exc}"
                )

    @property
    def file_ingestion_available(self) -> bool:
        return self._file_ingestion_available

    @property
    def file_ingestion_reason(self) -> str:
        return self._file_ingestion_reason

    def _build_agent(self) -> Agent:
        dog_fact_tool = ComponentTool(
            component=DogFactFetcher(),
            name="get_dog_fact",
            description=(
                "Retrieves a random interesting fact about dogs. Use for dog-related user requests."
            ),
        )
        dog_image_tool = ComponentTool(
            component=DogImageFetcher(
                openai_api_key=self.settings.openai_api_key,
                openai_base_url=self.settings.openai_base_url,
            ),
            name="get_dog_image_with_description",
            description=(
                "Fetches a random dog image and returns an AI-generated breed description."
            ),
        )
        weather_tool = ComponentTool(
            component=WeatherFetcher(),
            name="get_weather",
            description="Gets current weather for a city in Russian or English.",
        )

        agent = Agent(
            chat_generator=OpenAIChatGenerator(
                model=self.settings.chat_model,
                api_key=Secret.from_token(self.settings.openai_api_key),
                api_base_url=self.settings.openai_base_url,
            ),
            tools=[dog_fact_tool, dog_image_tool, weather_tool],
            system_prompt=(
                "You are a smart personal assistant helping users through Telegram.\n"
                "Use context when available, remain concise and conversational.\n"
                "Use tools only when user explicitly asks for relevant data.\n"
                "Do not suggest tools unless user asks. Answer naturally.\n"
                "If user writes in Russian, answer in Russian."
            ),
            max_agent_steps=10,
            exit_conditions=["text"],
        )
        agent.warm_up()
        return agent

    def _build_generation_pipeline(self) -> Pipeline:
        pipe = Pipeline()
        pipe.add_component("query_embedder", self._new_text_embedder())
        pipe.add_component("retriever", self._new_retriever())
        pipe.add_component("context_builder", RetrievedContextFormatter())
        pipe.add_component("responder", AgentResponder(self.agent))

        pipe.connect("query_embedder.embedding", "retriever.query_embedding")
        pipe.connect("retriever.documents", "context_builder.documents")
        pipe.connect("context_builder.prompt", "responder.prompt")
        return pipe

    def _build_conversation_pipeline(self) -> Pipeline:
        pipe = Pipeline()
        pipe.add_component("builder", ConversationDocumentBuilder())
        pipe.add_component("embedder", self._new_document_embedder())
        pipe.add_component("writer", self._new_writer())
        pipe.connect("builder.documents", "embedder.documents")
        pipe.connect("embedder.documents", "writer.documents")
        return pipe

    def _build_ingestion_pipeline(self) -> Pipeline:
        pipe = Pipeline()
        pipe.add_component(
            "converter",
            DoclingConverter(export_type=ExportType.MARKDOWN),
        )
        pipe.add_component(
            "splitter",
            DocumentSplitter(split_by="word", split_length=180, split_overlap=30),
        )
        pipe.add_component("enricher", FileMetadataEnricher())
        pipe.add_component("embedder", self._new_document_embedder())
        pipe.add_component("writer", self._new_writer())

        pipe.connect("converter.documents", "splitter.documents")
        pipe.connect("splitter.documents", "enricher.documents")
        pipe.connect("enricher.documents", "embedder.documents")
        pipe.connect("embedder.documents", "writer.documents")
        return pipe

    def _new_text_embedder(self) -> OpenAITextEmbedder:
        return OpenAITextEmbedder(
            model=self.settings.embedding_model,
            api_key=Secret.from_token(self.settings.openai_api_key),
            api_base_url=self.settings.openai_base_url,
        )

    def _new_document_embedder(self) -> OpenAIDocumentEmbedder:
        return OpenAIDocumentEmbedder(
            model=self.settings.embedding_model,
            api_key=Secret.from_token(self.settings.openai_api_key),
            api_base_url=self.settings.openai_base_url,
        )

    def _new_retriever(self) -> PineconeEmbeddingRetriever:
        return PineconeEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.settings.retriever_top_k,
        )

    def _new_writer(self) -> DocumentWriter:
        return DocumentWriter(document_store=self.document_store)

    @staticmethod
    def _configure_hf_windows_symlink_fallback() -> None:
        # On some Windows setups HF symlink creation fails with WinError 1314.
        # In this case we transparently copy files instead of creating symlinks.
        if os.name != "nt":
            return
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            from huggingface_hub import file_download
        except Exception:
            return

        if getattr(file_download, "_zc_symlink_patch_applied", False):
            return

        original_create_symlink = file_download._create_symlink

        def _safe_create_symlink(src: str, dst: str, new_blob: bool = False) -> None:
            try:
                original_create_symlink(src=src, dst=dst, new_blob=new_blob)
            except OSError as exc:
                if getattr(exc, "winerror", None) == 1314:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if os.path.lexists(dst):
                        os.remove(dst)
                    shutil.copy2(src, dst)
                    return
                raise

        file_download._create_symlink = _safe_create_symlink
        file_download._zc_symlink_patch_applied = True

    def generate_response(self, user_id: int, query: str) -> Tuple[str, Optional[str]]:
        result = self.generation_pipeline.run(
            {
                "query_embedder": {"text": query},
                "retriever": {
                    "filters": {"field": "user_id", "operator": "==", "value": str(user_id)},
                    "top_k": self.settings.retriever_top_k,
                },
                "context_builder": {"query": query},
            }
        )
        response_text = result["responder"]["response_text"]
        image_url = result["responder"]["image_url"]
        return response_text, image_url

    def store_conversation(
        self,
        user_id: int,
        username: str,
        user_message: str,
        assistant_response: str,
    ) -> None:
        self.conversation_pipeline.run(
            {
                "builder": {
                    "user_id": user_id,
                    "username": username,
                    "user_message": user_message,
                    "assistant_response": assistant_response,
                }
            }
        )

    def ingest_file(
        self,
        file_path: str,
        file_name: str,
        user_id: int,
        username: str,
    ) -> Dict[str, Any]:
        if not self.file_ingestion_available:
            return {
                "status": "unavailable",
                "reason": self.file_ingestion_reason,
            }

        return self.ingestion_pipeline.run(
            {
                "converter": {"paths": [file_path]},
                "enricher": {
                    "user_id": user_id,
                    "username": username,
                    "file_name": file_name,
                },
            }
        )

    def summarize_documents(self, file_name: str, documents: List[Document]) -> str:
        snippets = [doc.content for doc in documents[:5] if doc.content]
        joined = "\n\n".join(snippets)
        if len(joined) > 5000:
            joined = joined[:5000] + "..."

        prompt = (
            f"Ниже фрагменты файла '{file_name}'. "
            "Сформулируй ровно одно короткое предложение-резюме на русском языке.\n\n"
            f"{joined}"
        )
        result = self.summary_generator.run(messages=[ChatMessage.from_user(prompt)])
        replies = result.get("replies") or []
        first_reply = replies[0] if replies else ""
        if hasattr(first_reply, "text"):
            text = (first_reply.text or "").strip()
        else:
            text = str(first_reply).strip()
        return self._normalize_one_sentence(text)

    @staticmethod
    def _normalize_one_sentence(text: str) -> str:
        if not text:
            return "Файл обработан и содержит структурированную информацию для дальнейших ответов."
        for separator in [". ", "! ", "? "]:
            if separator in text:
                first = text.split(separator, maxsplit=1)[0].strip()
                ending = separator.strip()[0]
                return f"{first}{ending}"
        if text[-1] not in ".!?":
            return text + "."
        return text

