class Product:
    """Class representing a product."""
    
    def __init__(self, id: int, name: str, image_url: str, description: str, model: str, color: str, cloth_type: str):
        self.id: int = id
        self.name: str = name
        self.image_url: str = image_url
        self.description: str = description
        self.model: str = model
        self.color: str = color
        self.cloth_type: str = cloth_type

    def to_dict(self) -> dict:
        """Convert the product to a dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'image_url': self.image_url,
            'description': self.description,
            'model': self.model,
            'color': self.color,
            'cloth_type': self.cloth_type
        }