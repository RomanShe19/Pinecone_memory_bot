"""
Telegram Bot Assistant with Pinecone Memory

This bot uses Pinecone vector database to store and retrieve conversation history,
providing context-aware responses to users.

Note: Only real user messages and bot responses are saved to Pinecone.
Service commands (/start, /help, /stats, /clear) are not saved.
"""

import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
import telebot
from telebot import types
from dotenv import load_dotenv
from pinecone_manager import PineconeManager
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging - file handler with detailed logs
file_handler = logging.FileHandler('telegram_bot.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler with cleaner format
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Suppress verbose telebot and urllib3 logs in console
for log_name in ['TeleBot', 'urllib3', 'httpx', 'httpcore']:
    log = logging.getLogger(log_name)
    log.setLevel(logging.WARNING)
    log.propagate = False
    log.addHandler(file_handler)  # Still log to file

# Create custom logger for our bot
logger = logging.getLogger('BotApp')
logger.setLevel(logging.INFO)

# Initialize bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Initialize Pinecone Manager
pinecone_manager = PineconeManager()

# Initialize OpenAI client for chat completions
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Bot configuration
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_CONTEXT_MESSAGES = 10
BOT_PERSONALITY = """Ты дружелюбный и полезный AI-ассистент в Telegram. 
Отвечай кратко, по существу и дружелюбно. Помогай пользователям с их вопросами, 
используя контекст из предыдущих сообщений."""


def setup_pinecone_index():
    """Initialize or connect to Pinecone index"""
    try:
        index_name = os.getenv("PINECONE_INDEX_NAME", "telegram-bot-memory")
        existing_indexes = pinecone_manager.list_indexes()
        
        if index_name not in existing_indexes:
            logger.info(f"Creating new index: {index_name}")
            pinecone_manager.create_index(index_name=index_name)
        else:
            logger.info(f"Connecting to existing index: {index_name}")
            pinecone_manager.connect_to_index(index_name=index_name)
        
        logger.info("✅ Pinecone index ready!")
    except Exception as e:
        logger.error(f"❌ Error setting up Pinecone: {e}", exc_info=True)
        raise


def get_user_namespace(user_id: int) -> str:
    """Generate namespace for user's conversation history"""
    return f"user_{user_id}"


def save_message_to_pinecone(
    user_id: int,
    username: str,
    message_text: str,
    role: str = "user",
    message_id: int = None
):
    """Save a message to Pinecone with metadata"""
    try:
        namespace = get_user_namespace(user_id)
        timestamp = datetime.now().isoformat()
        
        # Create unique ID for the message
        doc_id = f"{user_id}_{role}_{int(time.time() * 1000)}_{message_id or ''}"
        
        # Build metadata, only include message_id if it's not None
        metadata = {
            "user_id": user_id,
            "username": username,
            "role": role,
            "timestamp": timestamp
        }
        
        # Only add message_id if it exists (not None)
        if message_id is not None:
            metadata["message_id"] = message_id
        
        # Save using smart upsert to avoid duplicates
        result = pinecone_manager.upsert_document_smart(
            id=doc_id,
            text=message_text,
            metadata=metadata,
            namespace=namespace,
            model=EMBEDDING_MODEL,
            check_duplicates=True,
            threshold=0.95
        )
        
        # Log result with details
        action_emoji = {
            "inserted": "✅",
            "skipped": "⏭️",
            "updated": "🔄"
        }
        emoji = action_emoji.get(result['action'], "📝")
        
        logger.info(f"{emoji} [{result['action'].upper()}] {role} message | ID: {doc_id}")
        logger.debug(f"   Reason: {result['reason']}")
        
        # Show duplicate info if available
        if result.get('duplicate_check') and result['duplicate_check'].get('is_duplicate'):
            similarity = result['duplicate_check']['max_similarity']
            logger.debug(f"   Similarity: {similarity:.4f}")
        
        return doc_id
    
    except Exception as e:
        logger.error(f"❌ Error saving message to Pinecone: {e}", exc_info=True)
        return None


def get_conversation_context(user_id: int, current_message: str, top_k: int = MAX_CONTEXT_MESSAGES) -> List[Dict[str, Any]]:
    """Retrieve relevant conversation history from Pinecone"""
    try:
        namespace = get_user_namespace(user_id)
        
        # Query similar messages from history
        results = pinecone_manager.query_by_text(
            text=current_message,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        # Format results
        context = pinecone_manager.format_results(results, include_scores=True)
        
        # Sort by timestamp to maintain conversation order
        context_sorted = sorted(
            context,
            key=lambda x: x.get("metadata", {}).get("timestamp", ""),
            reverse=False
        )
        
        return context_sorted
    
    except Exception as e:
        logger.error(f"❌ Error retrieving context: {e}", exc_info=True)
        return []


def format_context_for_llm(context: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert Pinecone context to OpenAI chat format"""
    messages = []
    
    for item in context:
        role = item.get("metadata", {}).get("role", "user")
        text = item.get("text", "")
        
        # Convert role to OpenAI format
        if role == "bot":
            role = "assistant"
        
        messages.append({
            "role": role,
            "content": text
        })
    
    return messages


def generate_response(user_message: str, context: List[Dict[str, Any]]) -> str:
    """Generate AI response using OpenAI with conversation context"""
    try:
        # Format context messages
        context_messages = format_context_for_llm(context)
        
        # Build messages for OpenAI
        messages = [
            {"role": "system", "content": BOT_PERSONALITY}
        ]
        
        # Add context (last N messages)
        if context_messages:
            messages.extend(context_messages[-MAX_CONTEXT_MESSAGES:])
        
        # Add current message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Generate response
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"❌ Error generating response: {e}", exc_info=True)
        return "Извините, произошла ошибка при генерации ответа. Попробуйте еще раз."


def get_user_stats(user_id: int) -> Dict[str, Any]:
    """Get statistics about user's conversation history"""
    try:
        namespace = get_user_namespace(user_id)
        stats = pinecone_manager.get_index_stats()
        
        user_stats = {
            "total_vectors": stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0),
            "namespace": namespace
        }
        
        return user_stats
    
    except Exception as e:
        logger.error(f"❌ Error getting user stats: {e}", exc_info=True)
        return {"total_vectors": 0, "namespace": "unknown"}


@bot.message_handler(commands=['start'])
def send_welcome(message):
    """Handle /start command - service command, not saved to Pinecone"""
    user_id = message.from_user.id
    username = message.from_user.username or message.from_user.first_name
    
    logger.info(f"🚀 /start from @{username} (ID: {user_id})")
    
    welcome_text = f"""👋 Привет, {username}!

Я AI-помощник с памятью на базе Pinecone. Я запоминаю наши разговоры и использую их для более персонализированных ответов.

Доступные команды:
/start - Показать это сообщение
/stats - Статистика разговоров
/clear - Очистить историю
/help - Помощь

Просто напиши мне что-нибудь, и я отвечу с учетом контекста! 💬"""
    
    bot.reply_to(message, welcome_text)


@bot.message_handler(commands=['stats'])
def send_stats(message):
    """Handle /stats command - service command, not saved to Pinecone"""
    user_id = message.from_user.id
    username = message.from_user.username or message.from_user.first_name
    
    logger.info(f"📊 /stats from @{username} (ID: {user_id})")
    
    stats = get_user_stats(user_id)
    logger.info(f"   User has {stats['total_vectors']} messages in history")
    
    stats_text = f"""📊 Статистика твоей истории:

💬 Всего сообщений: {stats['total_vectors']}
🗂 Namespace: {stats['namespace']}
🤖 Модель чата: {CHAT_MODEL}
🧠 Модель эмбеддингов: {EMBEDDING_MODEL}

Все твои сообщения надежно сохранены в векторной базе данных Pinecone!"""
    
    bot.reply_to(message, stats_text)


@bot.message_handler(commands=['clear'])
def clear_history(message):
    """Handle /clear command - service command, not saved to Pinecone"""
    user_id = message.from_user.id
    username = message.from_user.username or message.from_user.first_name
    
    logger.info(f"🗑️ /clear from @{username} (ID: {user_id})")
    
    # Create confirmation keyboard
    markup = types.InlineKeyboardMarkup()
    btn_yes = types.InlineKeyboardButton("✅ Да, очистить", callback_data=f"clear_yes_{user_id}")
    btn_no = types.InlineKeyboardButton("❌ Отмена", callback_data=f"clear_no_{user_id}")
    markup.add(btn_yes, btn_no)
    
    confirm_text = "⚠️ Вы уверены, что хотите удалить всю историю разговоров?\nЭто действие нельзя отменить!"
    
    bot.reply_to(message, confirm_text, reply_markup=markup)


@bot.callback_query_handler(func=lambda call: call.data.startswith('clear_'))
def handle_clear_callback(call):
    """Handle clear history confirmation"""
    action = call.data.split('_')[1]
    user_id = int(call.data.split('_')[2])
    
    logger.info(f"🔘 Clear callback: action={action}, user_id={user_id}")
    
    # Check if the callback is from the same user
    if call.from_user.id != user_id:
        bot.answer_callback_query(call.id, "❌ Это не ваша команда!")
        return
    
    if action == "yes":
        try:
            namespace = get_user_namespace(user_id)
            logger.info(f"   🗑️ Clearing namespace: {namespace}")
            pinecone_manager.delete_all(namespace=namespace)
            logger.info(f"   ✅ History cleared successfully")
            bot.edit_message_text(
                "✅ История разговоров успешно очищена!",
                call.message.chat.id,
                call.message.message_id
            )
            bot.answer_callback_query(call.id, "✅ Удалено!")
        except Exception as e:
            logger.error(f"   ❌ Error clearing history: {e}", exc_info=True)
            bot.edit_message_text(
                f"❌ Ошибка при очистке истории: {str(e)}",
                call.message.chat.id,
                call.message.message_id
            )
            bot.answer_callback_query(call.id, "❌ Ошибка!")
    else:
        logger.info(f"   ❌ User cancelled clearing history")
        bot.edit_message_text(
            "❌ Очистка истории отменена.",
            call.message.chat.id,
            call.message.message_id
        )
        bot.answer_callback_query(call.id, "Отменено")


@bot.message_handler(commands=['help'])
def send_help(message):
    """Handle /help command - service command, not saved to Pinecone"""
    user_id = message.from_user.id
    username = message.from_user.username or message.from_user.first_name
    
    logger.info(f"❓ /help from @{username} (ID: {user_id})")
    
    help_text = """🤖 Помощь по боту

Я умею:
• Запоминать наши разговоры (кроме команд)
• Использовать контекст из прошлых сообщений
• Отвечать на вопросы с учетом истории
• Вести персонализированные диалоги

Команды:
/start - Начать работу с ботом
/stats - Посмотреть статистику
/clear - Очистить историю разговоров
/help - Показать эту справку

Технологии:
• 🧠 Pinecone - векторная база данных
• 🤖 OpenAI GPT - генерация ответов
• 📊 Эмбеддинги - семантический поиск

Просто напиши мне сообщение, и я отвечу с учетом контекста наших разговоров! 💬"""
    
    bot.reply_to(message, help_text)


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Handle all text messages - saves both user messages and bot responses to Pinecone"""
    user_id = message.from_user.id
    username = message.from_user.username or message.from_user.first_name
    user_message = message.text
    
    logger.info(f"💬 Message from @{username}: {user_message[:50]}{'...' if len(user_message) > 50 else ''}")
    
    # Show typing indicator
    bot.send_chat_action(message.chat.id, 'typing')
    
    try:
        # Save user message to Pinecone
        logger.debug("[1/4] Saving user message...")
        save_message_to_pinecone(
            user_id=user_id,
            username=username,
            message_text=user_message,
            role="user",
            message_id=message.message_id
        )
        
        # Get conversation context
        logger.debug("[2/4] Retrieving conversation context...")
        context = get_conversation_context(user_id, user_message)
        logger.info(f"📚 Retrieved {len(context)} context messages")
        
        # Generate response
        logger.debug("[3/4] Generating AI response...")
        bot_response = generate_response(user_message, context)
        logger.debug(f"🤖 Response length: {len(bot_response)} characters")
        
        # Save bot response to Pinecone
        logger.debug("[4/4] Saving bot response...")
        save_message_to_pinecone(
            user_id=user_id,
            username="bot",
            message_text=bot_response,
            role="bot",
            message_id=None
        )
        
        # Send response to user
        bot.reply_to(message, bot_response)
        logger.info("✅ Message processed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error processing message: {e}", exc_info=True)
        bot.reply_to(message, "Извините, произошла ошибка. Попробуйте еще раз.")


def main():
    """Main function to run the bot"""
    logger.info("🚀 Starting Telegram Bot with Pinecone Memory...")
    
    try:
        # Setup Pinecone index
        setup_pinecone_index()
        
        # Start bot
        logger.info("✅ Bot is ready and listening for messages...")
        logger.info(f"📊 Chat Model: {CHAT_MODEL}")
        logger.info(f"🧠 Embedding Model: {EMBEDDING_MODEL}")
        logger.info("=" * 50)
        
        # Start polling with retry logic
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Start polling with longer timeout to reduce timeout errors
                bot.infinity_polling(timeout=30, long_polling_timeout=20, skip_pending=True)
                break  # If successful, exit retry loop
                
            except telebot.apihelper.ApiTelegramException as e:
                if "409" in str(e) or "Conflict" in str(e):
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Bot conflict detected (attempt {attempt + 1}/{max_retries})")
                        logger.info(f"   Another instance may be running. Waiting {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error("\n❌ Failed to start bot after multiple attempts.")
                        logger.error("   Please check:")
                        logger.error("   1. Close any other running instances of the bot")
                        logger.error("   2. Stop the bot in Telegram if it's stuck")
                        logger.error("   3. Wait 30 seconds and try again\n")
                        raise
                else:
                    raise  # Re-raise if it's not a conflict error
        
    except KeyboardInterrupt:
        logger.info("\n👋 Bot stopped by user")
    except Exception as e:
        # Don't log again if already logged
        if not isinstance(e, telebot.apihelper.ApiTelegramException):
            logger.error(f"❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
