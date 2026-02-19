# Telegram-боты с “памятью” (Pinecone) + Haystack Agent

Проект демонстрирует **долгосрочную память на Pinecone** (семантический поиск по истории диалога) и две реализации Telegram‑бота:

- **Базовая**: `telegram_bot.py` + `pinecone_manager.py` (прямые вызовы OpenAI + Pinecone, дедупликация, namespace на пользователя)
- **Агентная (Haystack)**: `Hay/hay-telegram-bot.py` (Haystack `Agent` + `ComponentTool`, инструменты: погода/факты/картинки собак, память через `PineconeDocumentStore`)

## Возможности

- **Долгосрочная память**: сообщения и ответы сохраняются в Pinecone как векторы
- **Контекстные ответы**: перед ответом извлекается релевантный контекст по косинусному сходству
- **Изоляция пользователей**
  - в базовой версии: namespace `user_{id}`
  - в Haystack‑версии: фильтрация по метаданным `user_id`
- **Инструменты (только когда уместно по контексту)**
  - **Погода**: Open‑Meteo + Nominatim (без API‑ключа)
  - **Факт о собаках**: `dogapi.dog`
  - **Картинка собаки + описание породы**: `dog.ceo` + OpenAI Vision

## Стек

- Python 3.10+ (рекомендуется)
- Telegram: `pyTelegramBotAPI`
- LLM: OpenAI (через `openai`) или совместимый endpoint (`OPENAI_BASE_URL`)
- Память: Pinecone
- Агент: Haystack (`haystack-ai`) + интеграция Pinecone (`pinecone-haystack`)

## Быстрый старт (Windows / PowerShell)

1) Создайте виртуальное окружение и установите зависимости:

```bash
py -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

2) Создайте `.env` из примера и заполните ключи:

```bash
copy .env.example .env
```

3) Подготовьте Pinecone index:

- **metric**: `cosine`
- **dimension**: `1536` (используются эмбеддинги OpenAI `text-embedding-3-small`)
- **name**: как в `PINECONE_INDEX_NAME` (по умолчанию `telegram-bot-memory`)

Если у вас уже есть индекс с другой размерностью — создайте новый, иначе будут ошибки вида “Vector dimension … does not match …”.

## Запуск

### Базовая версия (без Haystack)

```bash
python telegram_bot.py
```

- **Логи**: `telegram_bot.log`
- **Команды**: `/start`, `/help`, `/stats`, `/clear`

### Haystack‑агент (tool calling)

```bash
cd Hay
python hay-telegram-bot.py
```

- **Логи**: `Hay/bot.log` (создаётся в текущей папке запуска)
- **Команды**: `/start`, `/help`, `/clear`
- Подробнее про эту реализацию: `Hay/README.md`

## Конфигурация

Переменные окружения (см. `.env.example`):

- `TELEGRAM_BOT_TOKEN`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (опционально, если используете прокси/совместимый endpoint)
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`

## Структура проекта

```
VPg06/
├── Hay/
│   ├── hay-telegram-bot.py      # Haystack Agent + инструменты + PineconeDocumentStore
│   └── README.md                # Документация по Haystack-части
├── pinecone_manager.py          # Pinecone helper (базовая версия)
├── telegram_bot.py              # Базовый бот (OpenAI + PineconeManager)
├── requirements.txt
├── .env.example
└── .gitignore
```

