import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class to hold environment variables."""
    
    def __init__(self) -> None:
        self.telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN")
        self.api_base_url: str = os.getenv("API_BASE_URL")
        self.upload_endpoint: str = "/upload/"
        self.fastapi_upload_url: str = f"{self.api_base_url}{self.upload_endpoint}"

        if not self.telegram_bot_token or not self.api_base_url:
            raise EnvironmentError("Please set TELEGRAM_BOT_TOKEN and API_BASE_URL in the .env file.")