"""
Smart Personal Assistant Telegram Bot with Haystack and Pinecone
Features:
- Conversational memory using Pinecone vector database
- Tool calling for external APIs (dog facts, dog images)
- Context-aware responses using RAG approach
"""

import os
import logging
import requests
import base64
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

import telebot
from telebot import types

from haystack import Document, Pipeline, component
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.tools import ComponentTool
from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever

load_dotenv()

# Настройка логирования в файл
log_file = 'bot.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()  # Оставляем минимальный вывод в консоль
    ]
)

# Отключаем избыточные логи от библиотек
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('TeleBot').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@component
class DogFactFetcher:
    """Haystack компонент для получения случайных фактов о собаках"""
    
    @component.output_types(fact=str)
    def run(self) -> Dict[str, Any]:
        """Получить случайный факт о собаках из API"""
        try:
            response = requests.get('https://dogapi.dog/api/v2/facts', timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    fact = data['data'][0]['attributes']['body']
                    logger.info(f"🐕 Получен факт о собаках: {fact[:50]}...")
                    return {"fact": fact}
            return {"fact": "Не удалось получить факт о собаках в данный момент."}
        except Exception as e:
            logger.error(f"❌ Ошибка при получении факта о собаках: {e}")
            return {"fact": "Извините, не удалось получить факт о собаках."}


@component
class WeatherFetcher:
    """Haystack компонент для получения информации о погоде"""
    
    @component.output_types(weather_info=str)
    def run(self, location: str) -> Dict[str, Any]:
        """Получить информацию о погоде для указанного города"""
        try:
            # Используем Open-Meteo API - бесплатное и надежное API без ключа
            # Сначала получаем координаты города через Nominatim (OpenStreetMap)
            logger.info(f"🌤️ Запрос погоды для: {location}")
            
            # Геокодирование города
            geocode_url = f"https://nominatim.openstreetmap.org/search"
            geocode_params = {
                'q': location,
                'format': 'json',
                'limit': 1,
                'accept-language': 'ru'
            }
            headers = {'User-Agent': 'TelegramWeatherBot/1.0'}
            
            geo_response = requests.get(geocode_url, params=geocode_params, headers=headers, timeout=10)
            if geo_response.status_code != 200 or not geo_response.json():
                return {"weather_info": f"Не удалось найти город: {location}"}
            
            geo_data = geo_response.json()[0]
            lat = geo_data['lat']
            lon = geo_data['lon']
            city_name = geo_data.get('display_name', location).split(',')[0]
            
            logger.info(f"📍 Найдены координаты: {lat}, {lon}")
            
            # Получаем погоду через Open-Meteo
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                'latitude': lat,
                'longitude': lon,
                'current': 'temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,wind_direction_10m',
                'timezone': 'auto'
            }
            
            weather_response = requests.get(weather_url, params=weather_params, timeout=10)
            if weather_response.status_code != 200:
                return {"weather_info": f"Не удалось получить информацию о погоде для {location}."}
            
            weather_data = weather_response.json()
            current = weather_data.get('current', {})
            
            # Расшифровка кодов погоды WMO
            weather_codes = {
                0: "Ясно", 1: "Преимущественно ясно", 2: "Переменная облачность", 3: "Пасмурно",
                45: "Туман", 48: "Изморозь",
                51: "Легкая морось", 53: "Морось", 55: "Сильная морось",
                61: "Небольшой дождь", 63: "Дождь", 65: "Сильный дождь",
                71: "Небольшой снег", 73: "Снег", 75: "Сильный снегопад",
                80: "Ливень", 81: "Сильный ливень", 82: "Очень сильный ливень",
                95: "Гроза", 96: "Гроза с градом", 99: "Сильная гроза с градом"
            }
            
            temp = current.get('temperature_2m', 'N/A')
            feels_like = current.get('apparent_temperature', 'N/A')
            humidity = current.get('relative_humidity_2m', 'N/A')
            wind_speed = current.get('wind_speed_10m', 'N/A')
            wind_dir = current.get('wind_direction_10m', 'N/A')
            weather_code = current.get('weather_code', 0)
            weather_desc = weather_codes.get(weather_code, "Неизвестно")
            
            # Определяем направление ветра
            def get_wind_direction(degrees):
                if degrees is None or degrees == 'N/A':
                    return 'N/A'
                directions = ['С', 'СВ', 'В', 'ЮВ', 'Ю', 'ЮЗ', 'З', 'СЗ']
                idx = int((degrees + 22.5) / 45) % 8
                return directions[idx]
            
            wind_dir_text = get_wind_direction(wind_dir)
            
            weather_info = f"""Погода в {city_name}:
🌡️ Температура: {temp}°C (ощущается как {feels_like}°C)
☁️ Состояние: {weather_desc}
💧 Влажность: {humidity}%
💨 Ветер: {wind_speed} км/ч, направление {wind_dir_text}"""
            
            logger.info(f"✅ Погода получена для {city_name}")
            return {"weather_info": weather_info}
            
        except requests.exceptions.Timeout:
            logger.error(f"⏱️ Таймаут при получении погоды для {location}")
            return {"weather_info": f"Превышено время ожидания при получении погоды для {location}. Попробуйте позже."}
        except Exception as e:
            logger.error(f"❌ Ошибка при получении погоды: {e}")
            return {"weather_info": f"Извините, не удалось получить информацию о погоде для {location}."}


@component
class DogImageFetcher:
    """Haystack компонент для получения и описания изображений собак"""
    
    def __init__(self, openai_api_key: str, openai_base_url: str = None):
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
    
    @component.output_types(image_url=str, description=str)
    def run(self) -> Dict[str, Any]:
        """Получить случайное изображение собаки и описать его с помощью OpenAI Vision"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Получаем URL изображения
                response = requests.get('https://dog.ceo/api/breeds/image/random', timeout=15)
                if response.status_code != 200:
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Попытка {attempt + 1}/{max_retries} не удалась, повторяю...")
                        continue
                    return {
                        "image_url": "",
                        "description": "Не удалось получить изображение собаки в данный момент."
                    }
                
                data = response.json()
                if data['status'] != 'success':
                    if attempt < max_retries - 1:
                        continue
                    return {
                        "image_url": "",
                        "description": "Не удалось получить изображение собаки в данный момент."
                    }
                
                image_url = data['message']
                logger.info(f"🖼️ Получено изображение собаки: {image_url}")
                
                # Загружаем само изображение с увеличенным таймаутом
                img_response = requests.get(image_url, timeout=30)
                if img_response.status_code != 200:
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Не удалось загрузить изображение, попытка {attempt + 1}/{max_retries}")
                        continue
                    return {
                        "image_url": image_url,
                        "description": f"Найдено изображение, но не удалось загрузить его для описания."
                    }
                
                logger.info(f"✅ Изображение успешно загружено ({len(img_response.content)} байт)")
                image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                description = self._describe_dog_image(image_base64)
                
                return {
                    "image_url": image_url,
                    "description": description
                }
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"⏱️ Таймаут при загрузке изображения (попытка {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
                return {
                    "image_url": "",
                    "description": "Извините, не удалось загрузить изображение собаки из-за таймаута. Попробуйте ещё раз."
                }
            except Exception as e:
                logger.error(f"❌ Ошибка в DogImageFetcher (попытка {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
                return {
                    "image_url": "",
                    "description": "Извините, не удалось получить и описать изображение собаки."
                }
        
        return {
            "image_url": "",
            "description": "Извините, не удалось получить изображение собаки после нескольких попыток."
        }
    
    def _describe_dog_image(self, image_base64: str) -> str:
        """Использовать OpenAI Vision API для описания породы собаки"""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Проанализируй это изображение собаки и предоставь на русском языке:
1. Порода (или смесь пород, если применимо)
2. Краткая история породы (происхождение, предназначение)
3. Ключевые характеристики и черты
4. Интересные факты о породе

Будь информативным, но кратким (3-4 предложения)."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            description = response.choices[0].message.content
            logger.info(f"📝 Сгенерировано описание изображения: {description[:100]}...")
            return description
            
        except Exception as e:
            logger.error(f"❌ Ошибка при описании изображения через OpenAI: {e}")
            return "Это похоже на собаку, но не удалось сгенерировать детальное описание."


class TelegramAssistant:
    """Smart personal assistant with memory and tool calling capabilities"""
    
    def __init__(self):
        self.bot = telebot.TeleBot(os.getenv('TELEGRAM_BOT_TOKEN'))
        self.setup_pinecone()
        self.setup_agent()
        self.setup_handlers()
        
    def setup_pinecone(self):
        """Инициализация Pinecone для хранения контекста диалогов"""
        logger.info("🔧 Инициализация Pinecone хранилища...")
        
        # Используем модель text-embedding-3-small от OpenAI с размерностью 1536
        # которая соответствует размерности индекса в Pinecone
        self.document_store = PineconeDocumentStore(
            index=os.getenv('PINECONE_INDEX_NAME', 'telegram-bot-memory'),
            metric="cosine",
            dimension=1536,
            spec={
                "serverless": {
                    "region": os.getenv('PINECONE_REGION', 'us-east-1'),
                    "cloud": os.getenv('PINECONE_CLOUD', 'aws')
                }
            }
        )
        
        # Используем OpenAI embeddings для совместимости с размерностью 1536
        from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder
        
        openai_api_key = os.getenv('OPENAI_API_KEY')
        openai_base_url = os.getenv('OPENAI_BASE_URL')
        
        self.text_embedder = OpenAITextEmbedder(
            model="text-embedding-3-small",
            api_key=Secret.from_token(openai_api_key),
            api_base_url=openai_base_url
        )
        
        self.doc_embedder = OpenAIDocumentEmbedder(
            model="text-embedding-3-small",
            api_key=Secret.from_token(openai_api_key),
            api_base_url=openai_base_url
        )
        
        self.retriever = PineconeEmbeddingRetriever(
            document_store=self.document_store,
            top_k=5
        )
        
        logger.info("✅ Pinecone хранилище успешно инициализировано")
    
    def setup_agent(self):
        """Настройка Haystack агента с инструментами"""
        logger.info("🤖 Настройка агента с инструментами...")
        
        openai_api_key = os.getenv('OPENAI_API_KEY')
        openai_base_url = os.getenv('OPENAI_BASE_URL')
        
        # Инструмент для получения фактов о собаках
        dog_fact_component = DogFactFetcher()
        dog_fact_tool = ComponentTool(
            component=dog_fact_component,
            name="get_dog_fact",
            description="Retrieves a random interesting fact about dogs. Use this when the user asks about dogs, wants to learn something about dogs, or needs dog-related information."
        )
        
        # Инструмент для получения изображений собак с описанием
        dog_image_component = DogImageFetcher(
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url
        )
        dog_image_tool = ComponentTool(
            component=dog_image_component,
            name="get_dog_image_with_description",
            description="Fetches a random dog image and provides an AI-generated description of the dog breed, including its history and characteristics. Use this when the user wants to see a dog picture or learn about dog breeds."
        )
        
        # Инструмент для получения погоды
        weather_component = WeatherFetcher()
        weather_tool = ComponentTool(
            component=weather_component,
            name="get_weather",
            description="Gets current weather information for a specified location (city name). Use this when the user asks about weather, temperature, or weather conditions in any city. The location parameter should be the city name in Russian or English."
        )
        
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(
                model="gpt-4o-mini",
                api_key=Secret.from_token(openai_api_key),
                api_base_url=openai_base_url
            ),
            tools=[dog_fact_tool, dog_image_tool, weather_tool],
            system_prompt="""You are a smart personal assistant helping users through Telegram.

Your key responsibilities:
1. Remember and use context from previous conversations
2. Be helpful, friendly, and conversational like a real assistant
3. Answer questions naturally and provide relevant information
4. Maintain conversation continuity by referencing past interactions when relevant

Available tools and when to use them:
- get_weather: Use when the user asks about weather, temperature, or weather conditions in any city
- get_dog_fact: Use when the user asks about dogs or wants dog-related information
- get_dog_image_with_description: Use when the user wants to see a dog picture or learn about dog breeds

Guidelines for tool usage:
- ONLY use tools when the user explicitly asks for that type of information
- DO NOT mention or suggest using tools unless the user brings up relevant topics
- DO NOT end your responses with offers to use tools - let the conversation flow naturally
- When using weather tool, extract the city name from the user's question

General behavior:
- Be concise and natural in your responses
- Act as a real personal assistant, not a bot that constantly offers features
- When you retrieve context from previous conversations, incorporate it naturally
- Respond to what the user actually asks, don't try to steer the conversation to your tools
- Answer in Russian when the user writes in Russian""",
            max_agent_steps=10,
            exit_conditions=["text"]
        )
        
        self.agent.warm_up()
        logger.info("✅ Агент успешно настроен и готов к работе")
    
    def store_conversation(self, user_id: int, username: str, message: str, response: str):
        """Сохранить диалог в Pinecone для будущего извлечения контекста"""
        try:
            timestamp = datetime.now().isoformat()
            
            conversation_text = f"Пользователь ({username}): {message}\nАссистент: {response}"
            
            logger.info(f"📝 Создание документа для сохранения...")
            doc = Document(
                content=conversation_text,
                meta={
                    "user_id": str(user_id),
                    "username": username,
                    "timestamp": timestamp,
                    "user_message": message,
                    "assistant_response": response
                }
            )
            
            logger.info(f"🔢 Создание эмбеддингов для документа...")
            docs_with_embeddings = self.doc_embedder.run(documents=[doc])
            
            if docs_with_embeddings and "documents" in docs_with_embeddings:
                embedded_doc = docs_with_embeddings["documents"][0]
                logger.info(f"✅ Эмбеддинг создан (размерность: {len(embedded_doc.embedding) if embedded_doc.embedding else 0})")
                logger.info(f"📋 Метаданные: user_id={user_id}, username={username}, timestamp={timestamp}")
                
                logger.info(f"💾 Сохранение в Pinecone...")
                self.document_store.write_documents(docs_with_embeddings["documents"])
                logger.info(f"✅ Диалог успешно сохранён для пользователя {username} (ID: {user_id})")
            else:
                logger.error(f"❌ Не удалось создать эмбеддинги")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении диалога: {e}", exc_info=True)
    
    def retrieve_context(self, user_id: int, query: str, top_k: int = 5) -> List[Document]:
        """Извлечь релевантный контекст диалога из Pinecone"""
        try:
            logger.info(f"🔍 Создание эмбеддинга для запроса: '{query[:50]}...'")
            embedding_result = self.text_embedder.run(text=query)
            query_embedding = embedding_result["embedding"]
            logger.info(f"✅ Эмбеддинг запроса создан (размерность: {len(query_embedding)})")
            
            # Pinecone фильтры используют синтаксис: {"field": {"operator": value}}
            # Для Haystack Pinecone integration используем правильный синтаксис
            logger.info(f"🔎 Поиск в Pinecone для user_id={user_id}, top_k={top_k}")
            retrieval_result = self.retriever.run(
                query_embedding=query_embedding,
                top_k=top_k,
                filters={"field": "user_id", "operator": "==", "value": str(user_id)}
            )
            
            documents = retrieval_result.get("documents", [])
            logger.info(f"📚 Извлечено {len(documents)} контекстных документов для пользователя {user_id}")
            
            if documents:
                for i, doc in enumerate(documents[:3], 1):  # Показываем первые 3
                    logger.info(f"  📄 Документ {i}: {doc.content[:100]}...")
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Ошибка при извлечении контекста: {e}", exc_info=True)
            return []
    
    def generate_response(self, user_id: int, username: str, message: str) -> tuple[str, str]:
        """Сгенерировать ответ используя агента с контекстом из Pinecone"""
        try:
            context_docs = self.retrieve_context(user_id, message)
            
            context_text = ""
            if context_docs:
                context_text = "\n\nКонтекст предыдущих разговоров:\n"
                for doc in context_docs:
                    context_text += f"- {doc.content}\n"
            
            full_message = f"{message}{context_text}"
            
            chat_messages = [ChatMessage.from_user(full_message)]
            
            logger.info(f"🤔 Генерация ответа для пользователя {username}...")
            result = self.agent.run(messages=chat_messages)
            
            response_text = result["messages"][-1].text
            
            image_url = None
            if "image_url" in str(result.get("messages", [])):
                for msg in result.get("messages", []):
                    if hasattr(msg, 'meta') and msg.meta:
                        tool_output = msg.meta.get('tool_output', {})
                        if 'image_url' in tool_output and tool_output['image_url']:
                            image_url = tool_output['image_url']
                            break
            
            logger.info(f"✅ Ответ сгенерирован успешно")
            return response_text, image_url
            
        except Exception as e:
            logger.error(f"❌ Ошибка при генерации ответа: {e}")
            return "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте снова.", None
    
    def setup_handlers(self):
        """Setup Telegram bot message handlers"""
        
        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(message):
            welcome_text = """👋 Привет! Я твой умный персональный помощник!

Я могу помочь тебе с:
• Ответами на вопросы и общением
• Запоминанием наших предыдущих разговоров
• Информацией о погоде в любом городе 🌤️
• Интересными фактами о собаках 🐕
• Показом картинок собак с описанием породы 📸

Просто напиши мне сообщение, и я помогу!

Команды:
/start - Показать это приветствие
/help - Показать справку
/clear - Очистить историю разговоров"""
            
            self.bot.reply_to(message, welcome_text)
        
        @self.bot.message_handler(commands=['clear'])
        def clear_history(message):
            try:
                user_id = message.from_user.id
                logger.info(f"🗑️ Очистка истории для пользователя {user_id}")
                self.bot.reply_to(
                    message,
                    "Примечание: Из-за архитектуры Pinecone я не могу удалить историю отдельного пользователя. Однако я начну общение с чистого листа!"
                )
            except Exception as e:
                logger.error(f"❌ Ошибка в команде clear: {e}")
                self.bot.reply_to(message, "Извините, произошла ошибка.")
        
        @self.bot.message_handler(func=lambda message: True)
        def handle_message(message):
            try:
                user_id = message.from_user.id
                username = message.from_user.username or message.from_user.first_name or "Пользователь"
                user_message = message.text
                
                logger.info(f"📨 Получено сообщение от {username} (ID: {user_id}): {user_message}")
                
                self.bot.send_chat_action(message.chat.id, 'typing')
                
                response_text, image_url = self.generate_response(user_id, username, user_message)
                
                if image_url:
                    try:
                        logger.info(f"📤 Отправка изображения пользователю {username}")
                        self.bot.send_photo(message.chat.id, image_url, caption=response_text)
                    except Exception as e:
                        logger.error(f"❌ Ошибка при отправке фото: {e}")
                        self.bot.reply_to(message, f"{response_text}\n\nСсылка на изображение: {image_url}")
                else:
                    self.bot.reply_to(message, response_text)
                
                self.store_conversation(user_id, username, user_message, response_text)
                
            except Exception as e:
                logger.error(f"❌ Ошибка при обработке сообщения: {e}", exc_info=True)
                self.bot.reply_to(message, "Извините, произошла ошибка. Пожалуйста, попробуйте снова.")
    
    def run(self):
        """Запустить бота"""
        logger.info("🚀 Запуск Telegram бота...")
        print("🤖 Бот запущен и работает! Нажмите Ctrl+C для остановки.")
        print(f"📋 Логи сохраняются в файл: bot.log")
        
        # Увеличенные таймауты для стабильной работы
        self.bot.infinity_polling(
            timeout=60,  # Таймаут для long polling
            long_polling_timeout=60,  # Таймаут для получения обновлений
            skip_pending=True  # Пропускаем старые сообщения при запуске
        )


def main():
    """Главная точка входа"""
    required_env_vars = [
        'TELEGRAM_BOT_TOKEN',
        'OPENAI_API_KEY',
        'PINECONE_API_KEY'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"❌ Отсутствуют обязательные переменные окружения: {', '.join(missing_vars)}")
        print(f"\n❌ Ошибка: Отсутствуют обязательные переменные окружения: {', '.join(missing_vars)}")
        print("Пожалуйста, создайте файл .env на основе .env.example и заполните ваши API ключи.")
        return
    
    try:
        assistant = TelegramAssistant()
        assistant.run()
    except KeyboardInterrupt:
        logger.info("⏹️ Бот остановлен пользователем")
        print("\n👋 Бот остановлен.")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)
        print(f"\n❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    main()
