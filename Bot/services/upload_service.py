import httpx
import logging
import base64

class UploadService:
    """Service for handling file uploads."""
    
    def __init__(self, upload_url: str) -> None:
        self.upload_url: str = upload_url
    
    async def upload_files(self, user_photo_base64: str, user_photo_extension: str, product_image_base64: str, 
                           product_image_extension: str, product_info: dict) -> dict:
        """Upload user photo and product image to the server and return the response."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.upload_url, data={
                    "user_photo": user_photo_base64,
                    "user_photo_extension": user_photo_extension,
                    "product_image": product_image_base64,
                    "product_image_extension": product_image_extension,
                    **product_info
                })
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logging.error(f"An error occurred while uploading files: {e}")
            raise 
    
    async def fetch_product_image(self, product_image_url: str) -> str:
        """Fetch product image from the given URL and return it as a base64 string."""
        try:
            async with httpx.AsyncClient() as client:
                product_image_response = await client.get(product_image_url)
                product_image_response.raise_for_status()
                product_image_bytes = product_image_response.content
                return product_image_bytes
        except httpx.HTTPStatusError as e:
            logging.error(f"Failed to fetch product image: {e.response.status_code} - {e.response.text}")
            raise  
        except Exception as e:
            logging.error(f"An error occurred while fetching product image: {e}")
            raise 