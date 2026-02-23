import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    openai_api_key: str
    pinecone_api_key: str
    openai_base_url: str
    pinecone_index_name: str
    pinecone_region: str
    pinecone_cloud: str
    embedding_model: str
    chat_model: str
    retriever_top_k: int
    log_file: str
    download_dir: Path
    polling_timeout: int
    long_polling_timeout: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "telegram-bot-memory"),
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            retriever_top_k=int(os.getenv("RETRIEVER_TOP_K", "5")),
            log_file=os.getenv("BOT_LOG_FILE", "hay_v2_bot.log"),
            download_dir=Path(os.getenv("BOT_DOWNLOAD_DIR", "hay_v2_bot_tmp")),
            polling_timeout=int(os.getenv("BOT_POLLING_TIMEOUT", "60")),
            long_polling_timeout=int(os.getenv("BOT_LONG_POLLING_TIMEOUT", "60")),
        )

    def validate(self) -> None:
        required = {
            "TELEGRAM_BOT_TOKEN": self.telegram_bot_token,
            "OPENAI_API_KEY": self.openai_api_key,
            "PINECONE_API_KEY": self.pinecone_api_key,
            "OPENAI_BASE_URL": self.openai_base_url,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(f"Missing required environment variables: {missing_list}")

        if "proxyapi.ru" not in self.openai_base_url.lower():
            raise ValueError(
                "OPENAI_BASE_URL must point to proxyapi.ru endpoint "
                "(example: https://api.proxyapi.ru/openai/v1)."
            )

        if self.embedding_model != "text-embedding-3-small":
            raise ValueError(
                "EMBEDDING_MODEL must be 'text-embedding-3-small' "
                "to keep remote OpenAI small-text embeddings consistent."
            )

