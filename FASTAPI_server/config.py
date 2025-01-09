import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class to hold environment variables."""

    def __init__(self) -> None:
        self.model_name: str = os.getenv("MODEL_NAME")
        self.ht_token: str = os.getenv("HT_TOKEN")
        self.js_data_url: str = os.getenv("JSON_DATA_URL")
        if not self.model_name or not self.ht_token or not self.js_data_url:
            raise EnvironmentError("Please set MODEL_NAME, HT_TOKEN and JS_DATA_URL in the .env file.")