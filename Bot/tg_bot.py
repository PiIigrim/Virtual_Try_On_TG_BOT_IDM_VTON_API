from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from config import Config
from logger import setup_logger
from handlers.telegram_handler import TelegramHandler
from services.product_service import ProductService
from services.upload_service import UploadService

def main() -> None:
    """Main function to run the Telegram bot."""
    
    setup_logger()
    config = Config()

    application = ApplicationBuilder().token(config.telegram_bot_token).build()

    product_service = ProductService(config.api_base_url)  
    upload_service = UploadService(config.fastapi_upload_url)
    
    telegram_handler = TelegramHandler(product_service=product_service, 
                                       upload_service=upload_service, 
                                       base_url_api=config.api_base_url)

    application.add_handler(CommandHandler("start", telegram_handler.start_menu))
    application.add_handler(CommandHandler("help", telegram_handler.help_command))
    application.add_handler(MessageHandler(filters.PHOTO, telegram_handler.handle_photo))
    application.add_handler(CallbackQueryHandler(telegram_handler.handle_button_click))

    application.run_polling()

if __name__ == "__main__":
    main()