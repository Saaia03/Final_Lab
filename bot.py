import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import asyncio
import logging

# Настройка логов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
BOT_TOKEN = "8092584045:AAH-laCoHhIz-msrxTW_QE58kIAtadsicmI"
API_URL = "http://localhost:7860/api/get_analysis"  # Новый GET-эндпоинт

# Инициализация бота
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Обработчик /start
@dp.message(Command(commands=['start']))
async def start_cmd(message: types.Message):
    await message.reply(
        "Привет! Отправь мне текст, и я определю его настроение."
    )

# Обработчик текстовых сообщений
@dp.message()
async def analyze_text(message: types.Message):
    text = message.text
    try:
        async with aiohttp.ClientSession() as session:
            # GET-запрос с параметром text
            async with session.get(
                f"{API_URL}?text={text}",
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    sentiment = result.get("result", "Не удалось распознать")
                    await message.reply(sentiment)
                else:
                    error = await response.text()
                    logger.error(f"API Error: {error}")
                    await message.reply("Ошибка при анализе текста")

    except Exception as e:
        logger.error(f"Error: {e}")
        await message.reply(f"Произошла ошибка: {str(e)}")

# Запуск бота
async def main():
    logger.info("Starting bot...")
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped")