# hay_v2_bot - Модульный Telegram-ассистент (Haystack + Docling + Pinecone)

`hay_v2_bot` - это Telegram-бот второго поколения с модульной архитектурой, где ядро построено на пайплайнах Haystack.

Версия сохраняет все ключевые возможности предыдущего Haystack-бота и добавляет обработку документов (PDF, DOCX и др.) через Docling с векторной индексацией в Pinecone.

## Что умеет бот

- Обрабатывает обычные текстовые сообщения с учетом контекста
- Использует семантическую память из Pinecone (фильтрация по `user_id`)
- Поддерживает вызов инструментов через Haystack Agent:
  - погода
  - факты о собаках
  - изображение собаки + описание породы
- Принимает файлы и запускает ingestion-пайплайн:
  - конвертация через Docling
  - разбиение на чанки
  - эмбеддинг
  - сохранение чанков в Pinecone
- Отправляет одно короткое резюме после успешной обработки файла
- Использует OpenAI через `OPENAI_BASE_URL` (совместимо с прокси)

## Архитектура

```text
hay_v2_bot/
├── components/
│   ├── external_tools.py         # Инструменты: погода / факт / изображение собаки
│   └── pipeline_components.py    # Переиспользуемые кастомные компоненты
├── pipelines/
│   └── pipeline_manager.py       # Сборка и запуск всех пайплайнов
├── bot/
│   └── telegram_bot.py           # Telegram-обработчики и runtime бота
├── config.py                     # Настройки из окружения + валидация
├── logging_setup.py              # Политика логирования (консоль/файл)
├── main.py                       # Точка входа
└── README.md
```

## Пайплайны

### 1) Generation pipeline

`query_embedder -> retriever -> context_builder -> responder`

- строит эмбеддинг входного текста
- извлекает релевантные фрагменты памяти/документов из Pinecone
- формирует промпт с контекстом
- получает финальный ответ от Haystack Agent

### 2) Conversation pipeline

`builder -> embedder -> writer`

- собирает `Document` из пары сообщение пользователя + ответ ассистента
- строит эмбеддинг через OpenAI-модель
- сохраняет документ в Pinecone для дальнейшего контекста

### 3) Ingestion pipeline (файлы)

`DoclingConverter(MARKDOWN) -> splitter -> enricher -> embedder -> writer`

- конвертирует файл через Docling
- разбивает контент на чанки
- обогащает метаданные (`file_name`, `chunk_index`, опционально `page_number`, `user_id`)
- эмбеддит и сохраняет чанки в Pinecone

## Требования

- Python 3.10+
- Зависимости из корневого `requirements.txt`
- Pinecone index с параметрами:
  - metric: `cosine`
  - dimension: `1536`

## Переменные окружения

Обязательные:

- `TELEGRAM_BOT_TOKEN`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (должен указывать на ваш прокси-эндпоинт, например `https://api.proxyapi.ru/openai/v1`)
- `PINECONE_API_KEY`

Часто используемые опциональные:

- `PINECONE_INDEX_NAME` (по умолчанию: `telegram-bot-memory`)
- `PINECONE_REGION` (по умолчанию: `us-east-1`)
- `PINECONE_CLOUD` (по умолчанию: `aws`)
- `CHAT_MODEL` (по умолчанию: `gpt-4o-mini`)
- `EMBEDDING_MODEL` (должен быть `text-embedding-3-small`)

## Запуск

Из корня проекта:

```bash
.\venv\Scripts\python.exe -m hay_v2_bot.main
```

Альтернативный вариант:

```bash
cd hay_v2_bot
python main.py
```

## Логирование

- Консоль: краткие рабочие логи в человекочитаемом виде
- Файл: полные логи, включая подробные traceback

Файл логов по умолчанию:

- `hay_v2_bot.log` (в текущей директории запуска)

## Диагностика проблем

- Если обработка файлов недоступна, проверьте:
  - `docling` и `docling-haystack` установлены в тот же `venv`, из которого запущен бот
  - стартовые логи:
    - `Python executable: ...`
    - `Docling ingestion available: yes/no`
- Если появляются таймауты Telegram API, это обычно проблема сети; бот обрабатывает сбои `send_chat_action` более мягко.
- На Windows проблемы прав на symlink в Hugging Face обрабатываются runtime-fallback логикой.

