import logging
import os
import base64
import httpx
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from services.product_service import ProductService
from services.upload_service import UploadService
from telegram.constants import ParseMode
from helpers.telegram_helpers import escape_special_chars

class TelegramHandler:
    """Handler for managing Telegram bot interactions."""
    
    def __init__(self, product_service: ProductService, upload_service: UploadService, base_url_api: str) -> None:
        self.product_service = product_service
        self.upload_service = upload_service
        self.base_url_api = base_url_api
        self.products = []

        self.command_map = {
            'start': self.start_menu,
            'next_product': self.next_product,
            'previous_product': self.previous_product,
            'select_product': self.select_product,
            'help': self.help_command,
            'how_to_send_photo': self.how_to_send_photo,
            'list_of_products': self.handle_list_of_products,
            'return_to_menu': self.start_menu,
            'show_catalog':self.show_catalog
        }

    async def start_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Initializes the start menu for the Telegram bot of the online store assistant."""
        if not self.products:
            self.products = await self.product_service.fetch_products()

        context.user_data['current_product_index'] = 0

        if not self.products:
            if update.message:
                await update.message.reply_text("❌ Ассортимент не найдены. Пожалуйста, попробуйте позже.")
            else:
                await update.callback_query.message.reply_text("❌ Ассортимент не найдены. Пожалуйста, попробуйте позже.")
            return 

        welcome_text = (
            "👋 Привет! Я ваш помощник в онлайн-магазине.\n"
            "С помощью меня вы можете просматривать товары и отправлять свои фото для обработки.\n"
            "Нажмите |Начать|, чтобы начать, или |Помощь|, если у вас есть вопросы."
        )

        keyboard = self.get_main_menu_keyboard()
        await self.send_message(update, welcome_text, keyboard)

    async def send_message(self, update: Update, text: str, reply_markup=None):
        """Sends a message to the user, either in response to a callback query or as a regular message."""
        if update.callback_query:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(text, reply_markup=reply_markup)
        else:
            await update.message.reply_text(text, reply_markup=reply_markup)

    def get_main_menu_keyboard(self):
        """Returns the main menu keyboard."""
        keyboard = [
            [InlineKeyboardButton("🔄 Начать", callback_data='show_catalog')],
            [InlineKeyboardButton("❓ Помощь", callback_data='help')]
        ]
        return InlineKeyboardMarkup(keyboard)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handles the help command by providing information about the bot's functionalities and sending a help message with options."""
        keyboard = self.get_help_menu_keyboard()
        help_text = (
            "👋 Привет! Я ваш помощник в онлайн-магазине.\n"
            "Вот что я могу:\n"
            "1. Просмотр товаров.\n"
            "2. Отправка фотографий для обработки.\n"
            "3. Получение помощи по использованию бота."
        )
        await self.send_message(update, help_text, keyboard)

    def get_help_menu_keyboard(self):
        """Creates and returns the help menu keyboard for the Telegram bot."""
        keyboard = [
            [InlineKeyboardButton("🔙 В меню", callback_data='return_to_menu')],
            [InlineKeyboardButton("📸 Как отправить изображение?", callback_data='how_to_send_photo')],
            [InlineKeyboardButton("🛍️ Список товаров", callback_data='list_of_products')],
        ]
        return InlineKeyboardMarkup(keyboard)

    async def how_to_send_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Instructs the user on how to send a photo for processing."""
        await self.send_message(update, "📸 Чтобы отправить изображение, выберите продукт и отправьте фото в формате JPG или PNG.")

    async def handle_list_of_products(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handles the request to list available products."""
        if not self.products:
            await self.send_message(update, "❌ Не удалось получить список товаров.")
            return
        
        product_list = "\n".join([f"{i+1}. {product.name}" for i, product in enumerate(self.products)])
        await self.send_message(update, f"🛍️ Доступные товары:\n{product_list}")

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles the reception and processing of a user's photo."""
        await self.send_message(update, "⏳ Получаю ваше изображение...")
        
        current_index = context.user_data.get('current_product_index', 0)
        selected_product = self.products[current_index]

        if not selected_product:
            await self.send_message(update, "❌ Сначала выберите продукт.")
            return
        
        try:
            user_photo_file = await update.message.photo[-1].get_file()
            user_photo_bytes = await user_photo_file.download_as_bytearray()
            user_photo_extension = os.path.splitext(user_photo_file.file_path)[1].lower()

            if user_photo_extension not in ['.png', '.jpg', '.jpeg']:
                await self.send_message(update, "❌ Пожалуйста, загрузите фото в формате JPG или PNG.")
                return
            
            product_info = {'product_description': selected_product.description}
            product_image_bytes = product_image_base64 = await self.upload_service.fetch_product_image(selected_product.image_url)

            user_photo_base64 = base64.b64encode(user_photo_bytes).decode('utf-8')
            product_image_base64 = base64.b64encode(product_image_bytes).decode('utf-8')
            response = await self.upload_service.upload_files(user_photo_base64, user_photo_extension, product_image_base64, 
                                                              os.path.splitext(selected_product.image_url)[1], product_info)
            
            if response and response.get('task_id'):
                await self.send_message(update, "✅ Файл загружен. Начинаю обработку...")
                await self.poll_status(update, response['task_id'], context)
            else:
                await self.send_message(update, "❌ Ошибка при загрузке файла.")
        
        except Exception as e:
            logging.error(f"Error: {e}")
            await self.send_message(update, "❌ Произошла ошибка. Попробуйте снова.")

    async def poll_status(self, update: Update, task_id, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Polls the status of the image processing task and updates the user on the progress."""
        processing = True
        while processing:
            await asyncio.sleep(12)
            try:
                async with httpx.AsyncClient() as client:
                    status_response = await client.get(f"{self.base_url_api}/status/{task_id}")
                    status_response.raise_for_status()
                    status_data = status_response.json()

                    if status_data['status'] == 'completed':
                        processed_image_base64 = status_data['result']
                        img_bytes = base64.b64decode(processed_image_base64)
                        await update.message.reply_photo(photo=img_bytes)
                        await asyncio.sleep(3)
                        await self.send_message(update, "✅ Status: Обработка завершена!")
                        await self.show_catalog(update, context)
                        processing = False

                    elif status_data['status'] == 'error':
                        await self.send_message(update, "❌ Status: Ошибка на стороне IDM-VTON API. Повторите позже.")
                        await asyncio.sleep(3)
                        await self.show_catalog(update, context)
                        processing = False
                    else:
                        await self.send_message(update, "⏳ Status: в обоработке...")

            except httpx.RequestError as e:
                logging.error(f"Request error while checking status for task {task_id}: {e}")
                await self.send_message(update, "❌ Status: Ошибка при получении статуса. Повторите позже.")
                processing = False
            except Exception as e:
                logging.error(f"Unexpected error for task {task_id}: {e}")
                await self.send_message(update, "❌ Status: Неизвестная ошибка. Повторите позже.")
                processing = False
        
    async def show_catalog(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Displays the current product from the catalog to the user."""
        current_index = context.user_data.get('current_product_index', 0)

        if current_index < 0 or current_index >= len(self.products):
            await self.send_message(update, "❌ Продукт не найден.")
            return

        product = self.products[current_index]
        product_text = (
            f"🛍️ *Название:* {escape_special_chars(product.name)}\n"
            f"🆔 *Модель:* {escape_special_chars(product.model)}\n"
            f"🎨 *Цвет:* {escape_special_chars(product.color)}\n"
            f"📜 *Описание:* {escape_special_chars(product.description)}\n"
        )
        
        keyboard = self.get_product_keyboard()

        if update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_photo(
                photo=product.image_url,
                caption=product_text,
                reply_markup=keyboard,
                parse_mode=ParseMode.MARKDOWN_V2
            )
        else:
            await update.message.reply_photo(
                photo=product.image_url,
                caption=product_text,
                reply_markup=keyboard,
                parse_mode=ParseMode.MARKDOWN_V2
            )

    def get_product_keyboard(self):
        """Creates and returns the inline keyboard for product navigation in the Telegram bot."""
        keyboard = [
            [InlineKeyboardButton("◀️ Назад", callback_data='previous_product'),
             InlineKeyboardButton("▶️ Вперед", callback_data='next_product')],
            [InlineKeyboardButton("✅ *Выбрать*", callback_data='select_product'),
             InlineKeyboardButton("🔙 В меню", callback_data='return_to_menu')]
        ]
        return InlineKeyboardMarkup(keyboard)

    async def next_product(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Advances the current product index to the next product in the list and displays it."""
        context.user_data['current_product_index'] = (context.user_data['current_product_index'] + 1) % len(self.products)
        await self.show_catalog(update, context)

    async def previous_product(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Moves the current product index to the previous product in the list and displays it."""
        context.user_data['current_product_index'] = (context.user_data['current_product_index'] - 1) % len(self.products)
        await self.show_catalog(update, context)

    async def select_product(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Selects the current product and prompts the user to send a photo for processing."""
        product = self.products[context.user_data['current_product_index']]
        await self.send_message(update, f"✅ Вы выбрали: {product.name}.\n*Теперь отправьте фото в jpeg/jpg/png*")

    async def handle_button_click(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles button click events from the inline keyboard and routes to the appropriate command."""
        query = update.callback_query.data
        await update.callback_query.answer()

        command = self.command_map.get(query)
        if command:
            await command(update, context)
        else:
            await self.send_message(update, "❌ Неизвестная команда.")