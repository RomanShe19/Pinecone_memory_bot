from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import telebot

from hay_v2_bot.config import Settings
from hay_v2_bot.pipelines.pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)


class TelegramAssistantV2:
    """Telegram bot wrapper around modular Haystack pipelines."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.bot = telebot.TeleBot(settings.telegram_bot_token)
        self.pipelines = PipelineManager(settings=settings)
        logger.info("Python executable: %s", sys.executable)
        logger.info(
            "Docling ingestion available: %s",
            "yes" if self.pipelines.file_ingestion_available else "no",
        )
        if not self.pipelines.file_ingestion_available:
            logger.warning("Docling ingestion reason: %s", self.pipelines.file_ingestion_reason)
        self.settings.download_dir.mkdir(parents=True, exist_ok=True)
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self.bot.message_handler(commands=["start", "help"])
        def send_welcome(message):
            welcome_text = (
                "👋 Привет! Я твой умный персональный помощник v2.\n\n"
                "Я умею:\n"
                "• Отвечать на вопросы с учетом контекста\n"
                "• Помнить предыдущие диалоги\n"
                "• Давать информацию о погоде 🌤️\n"
                "• Рассказывать факты о собаках 🐕\n"
                "• Показывать картинки собак с описанием породы 📸\n"
                "• Принимать файлы (PDF, DOCX и др.), анализировать и обсуждать их 📄\n\n"
                "Команды:\n"
                "/start - приветствие\n"
                "/help - справка\n"
                "/clear - очистить контекст общения"
            )
            self._safe_reply(message, welcome_text)

        @self.bot.message_handler(commands=["clear"])
        def clear_history(message):
            try:
                user_id = message.from_user.id
                logger.info("Clear requested for user_id=%s", user_id)
                self._safe_reply(
                    message,
                    (
                        "Примечание: из-за архитектуры Pinecone нельзя удалить историю "
                        "одного пользователя точечно. Я начну общение с чистого листа."
                    ),
                )
            except Exception as exc:
                logger.error("Error in /clear: %s", exc, exc_info=True)
                self._safe_reply(message, "Извините, произошла ошибка.")

        @self.bot.message_handler(content_types=["document"])
        def handle_document(message):
            user_id = message.from_user.id
            username = message.from_user.username or message.from_user.first_name or "Пользователь"
            file_name = message.document.file_name or f"file_{message.document.file_id}"
            temp_file_path: Optional[Path] = None

            try:
                if not self.pipelines.file_ingestion_available:
                    logger.warning("Docling ingestion disabled: missing dependencies")
                    self._safe_reply(
                        message,
                        self.pipelines.file_ingestion_reason,
                    )
                    return

                logger.debug("Document received from %s (%s): %s", username, user_id, file_name)
                self._safe_reply(
                    message,
                    "Файл получен. Запускаю анализ и сохранение. Это может занять немного времени…",
                )
                self._safe_send_chat_action(message.chat.id, "typing")

                file_info = self.bot.get_file(message.document.file_id)
                downloaded_file = self.bot.download_file(file_info.file_path)

                user_dir = self.settings.download_dir / str(user_id)
                user_dir.mkdir(parents=True, exist_ok=True)
                temp_file_path = user_dir / file_name
                with open(temp_file_path, "wb") as file_obj:
                    file_obj.write(downloaded_file)

                result = self.pipelines.ingest_file(
                    file_path=str(temp_file_path),
                    file_name=file_name,
                    user_id=user_id,
                    username=username,
                )
                if result.get("status") == "unavailable":
                    logger.warning("Document ingestion unavailable: %s", result.get("reason"))
                    self._safe_reply(message, str(result.get("reason")))
                    return
                indexed_docs = result.get("enricher", {}).get("documents", [])
                summary = self.pipelines.summarize_documents(file_name=file_name, documents=indexed_docs)

                self._safe_reply(
                    message,
                    "Готово. Я изучил этот файл, теперь можем его обсудить.",
                )
                self._safe_reply(message, summary)
            except Exception as exc:
                logger.error("Document processing failed: %s", exc, exc_info=True)
                error_text = str(exc).strip()
                if error_text:
                    self._safe_reply(message, f"Не удалось обработать файл: {error_text}")
                else:
                    self._safe_reply(
                        message,
                        "Не удалось обработать файл. Попробуйте еще раз или отправьте другой формат.",
                    )
            finally:
                if temp_file_path and temp_file_path.exists():
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        logger.warning("Temporary file was not removed: %s", temp_file_path)

        @self.bot.message_handler(content_types=["text"])
        def handle_text(message):
            try:
                user_id = message.from_user.id
                username = message.from_user.username or message.from_user.first_name or "Пользователь"
                user_text = message.text or ""

                logger.debug("Text message from %s (%s): %s", username, user_id, user_text)
                self._safe_send_chat_action(message.chat.id, "typing")

                response_text, image_url = self.pipelines.generate_response(
                    user_id=user_id,
                    query=user_text,
                )

                if image_url:
                    try:
                        self.bot.send_photo(message.chat.id, image_url, caption=response_text)
                    except Exception as exc:
                        logger.error("Photo send failed: %s", exc, exc_info=True)
                        self._safe_reply(
                            message, f"{response_text}\n\nСсылка на изображение: {image_url}"
                        )
                else:
                    self._safe_reply(message, response_text)

                self.pipelines.store_conversation(
                    user_id=user_id,
                    username=username,
                    user_message=user_text,
                    assistant_response=response_text,
                )
            except Exception as exc:
                logger.error("Text handling failed: %s", exc, exc_info=True)
                self._safe_reply(
                    message,
                    "Извините, произошла ошибка при обработке запроса. Пожалуйста, попробуйте снова.",
                )

    def run(self) -> None:
        logger.info("Starting Telegram bot v2")
        print("Bot v2 запущен. Нажмите Ctrl+C для остановки.")
        print(f"Логи: {self.settings.log_file}")

        self.bot.infinity_polling(
            timeout=self.settings.polling_timeout,
            long_polling_timeout=self.settings.long_polling_timeout,
            skip_pending=True,
        )

    def _safe_send_chat_action(self, chat_id: int, action: str) -> None:
        try:
            self.bot.send_chat_action(chat_id, action)
        except Exception as exc:
            logger.warning("Не удалось отправить chat action (%s): %s", action, exc)

    def _safe_reply(self, message, text: str) -> None:
        try:
            self.bot.reply_to(message, text)
        except Exception as exc:
            logger.error("Не удалось отправить сообщение пользователю: %s", exc)

