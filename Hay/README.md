# Haystack‑агент (Telegram бот)

Эта папка содержит **агентную версию** Telegram‑бота на Haystack: `hay-telegram-bot.py`.

- **Память**: Pinecone (`PineconeDocumentStore`) + семантический поиск по истории
- **Инструменты (tool calling)**: погода, факты о собаках, картинка собаки + описание породы
- **Логи**: пишутся в `bot.log` (создаётся рядом с файлом при запуске из этой папки)

Полная документация по проекту и альтернативная “базовая” реализация лежит в корневом `README.md`.

## Запуск

Из корня репозитория:

```bash
.\venv\Scripts\activate
cd Hay
python hay-telegram-bot.py
```

## Переменные окружения

Бот читает `.env` из корня репозитория (python‑dotenv ищет файл в родительских папках).

Минимально нужны:

- `TELEGRAM_BOT_TOKEN`
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` (index в Pinecone должен быть **dimension=1536**, metric **cosine**)
- `OPENAI_BASE_URL` (опционально)

Шаблон: `.env.example` в корне репозитория.

## Инструменты

- **get_weather**: Open‑Meteo (forecast) + Nominatim (геокодинг)
- **get_dog_fact**: `dogapi.dog`
- **get_dog_image_with_description**: `dog.ceo` + OpenAI Vision (описание на русском)
