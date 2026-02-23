from __future__ import annotations

import sys
from pathlib import Path

# Allow running both:
# 1) python -m hay_v2_bot.main
# 2) python main.py (from hay_v2_bot directory)
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from hay_v2_bot.bot.telegram_bot import TelegramAssistantV2
from hay_v2_bot.config import Settings
from hay_v2_bot.logging_setup import configure_logging


def main() -> None:
    settings = Settings.from_env()
    logger = configure_logging(settings.log_file)

    try:
        settings.validate()
    except ValueError as exc:
        logger.error(str(exc))
        print(f"\nОшибка: {exc}")
        print("Проверьте .env и заполните обязательные ключи.")
        return

    try:
        assistant = TelegramAssistantV2(settings=settings)
        assistant.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        print("\nБот остановлен.")
    except Exception as exc:
        logger.error("Critical bot error: %s", exc, exc_info=True)
        print(f"\nКритическая ошибка: {exc}")


if __name__ == "__main__":
    main()

