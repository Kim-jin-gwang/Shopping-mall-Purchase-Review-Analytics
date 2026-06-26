from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any

class BaseCrawler(ABC):
    """
    Abstract base class for shopping mall review crawlers.
    To support a new shopping mall, inherit from this class and implement all abstract methods.
    """

    @abstractmethod
    def search_products(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Searches for products related to the keyword.
        
        Args:
            keyword (str): The search keyword (e.g., 'jeans').
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing:
                - 'id': Unique product identifier (str)
                - 'brand': Brand name of the product (str)
                - 'name': Name of the product (str)
        """
        pass

    @abstractmethod
    def crawl_reviews(self, product_id: str, limit: int, has_photo: bool = False) -> Tuple[List[int], List[str]]:
        """
        Crawls reviews for a specific product ID.
        
        Args:
            product_id (str): The ID of the product.
            limit (int): The number of reviews to collect.
            has_photo (bool): If True, collects photo/style reviews. If False, general reviews.
            
        Returns:
            Tuple[List[int], List[str]]: A tuple containing:
                - List of review ratings (ints)
                - List of review contents (strings)
        """
        pass

    @abstractmethod
    def download_product_image(self, product_id: str, output_path: str) -> str:
        """
        Downloads the representative image of the product.
        
        Args:
            product_id (str): The ID of the product.
            output_path (str): File path where the image should be saved.
            
        Returns:
            str: Path to the saved image file or empty string if failed.
        """
        pass
