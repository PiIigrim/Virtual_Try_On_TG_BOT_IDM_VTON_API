import httpx
import logging
from models.product import Product

class ProductService:
    """Service for managing products."""
    
    def __init__(self, base_url: str) -> None:
        self.base_url: str = base_url

    async def fetch_products(self) -> list[Product]:
        """Fetch products from the API and return them as a list of Product objects."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/products/")
                response.raise_for_status()
                return [Product(**item) for item in response.json()]
        except Exception as e:
            logging.error("Error fetching products from API: %s", e)
            return []