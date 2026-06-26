import os
import re
import pandas as pd
from typing import Dict, Any, List

from .crawler.base import BaseCrawler
from .analyzer.sentiment import SentimentModel
from .visualization.plot import Visualizer

class AnalysisPipeline:
    """
    Orchestrates the entire flow of searching products, crawling reviews, 
    running sentiment analysis, and generating visualization charts.
    """
    
    def __init__(self, crawler: BaseCrawler, model: SentimentModel, visualizer: Visualizer):
        self.crawler = crawler
        self.model = model
        self.visualizer = visualizer

    def search(self, keyword: str) -> List[Dict[str, Any]]:
        """Searches for products and returns search results."""
        print(f"[*] Searching for products with keyword: '{keyword}'...")
        products = self.crawler.search_products(keyword)
        return products

    def run(self, keyword: str, selected_product: Dict[str, Any], limit: int, out_dir: str = 'data') -> Dict[str, Any]:
        """
        Executes the analysis pipeline for the selected product:
        1. Download product representation image
        2. Crawl general and photo reviews
        3. Save reviews in CSV
        4. Predict sentiment and analyze keywords
        5. Save visual reports (pie and bar charts)
        """
        product_id = selected_product['id']
        product_name = selected_product['name']
        brand_name = selected_product['brand']
        
        # Clean product name for file path
        cleaned_product_name = re.sub(r'[\/:*?"<>|]', '_', product_name).strip()
        
        product_dir = os.path.join(out_dir, keyword)
        os.makedirs(product_dir, exist_ok=True)
        
        print(f"\n[*] Processing analysis for product: [{brand_name}] {product_name} (ID: {product_id})")
        
        # 1. Download Product Image
        image_path = os.path.join(product_dir, f"{cleaned_product_name}.jpg")
        self.crawler.download_product_image(product_id, image_path)
        
        # 2. Crawl Reviews
        print("\n[*] Crawling General Reviews...")
        general_scores, general_texts = self.crawler.crawl_reviews(product_id, limit, has_photo=False)
        
        print("\n[*] Crawling Photo/Style Reviews...")
        photo_scores, photo_texts = self.crawler.crawl_reviews(product_id, limit, has_photo=True)
        
        # 3. Create DataFrames & Save to CSV
        print("\n[*] Saving crawled reviews to CSV files...")
        
        # Save individual CSVs
        df_general = pd.DataFrame({'별점': general_scores, '리뷰내용': general_texts})
        df_general.to_csv(os.path.join(product_dir, f"{cleaned_product_name}_general.csv"), encoding="utf-8-sig", index=True)
        
        df_photo = pd.DataFrame({'별점': photo_scores, '리뷰내용': photo_texts})
        df_photo.to_csv(os.path.join(product_dir, f"{cleaned_product_name}_photo.csv"), encoding="utf-8-sig", index=True)
        
        # Combine all and save final CSV
        combined_scores = general_scores + photo_scores
        combined_texts = general_texts + photo_texts
        df_final = pd.DataFrame({'별점': combined_scores, '리뷰내용': combined_texts})
        final_csv_path = os.path.join(product_dir, f"{cleaned_product_name}_final.csv")
        df_final.to_csv(final_csv_path, encoding="utf-8-sig", index=True)
        print(f"[+] Saved combined reviews to: {final_csv_path}")
        
        # 4. Sentiment Analysis
        # Ensure model is loaded
        if self.model.model is None:
            if not self.model.load():
                print("[!] Pre-trained model not found. Training a new model...")
                self.model.train()
                
        print("\n[*] Performing sentiment classification on reviews...")
        analysis_result = self.model.analyze_reviews(combined_texts)
        
        print(f"[+] Classification results: {analysis_result['positive_count']} positive, {analysis_result['negative_count']} negative reviews")
        
        # 5. Visualization & Charts
        print("\n[*] Generating visualization charts...")
        pie_chart_path = os.path.join(product_dir, f"{cleaned_product_name}_sentiment_ratio.png")
        self.visualizer.plot_sentiment_ratio(
            positive_count=analysis_result['positive_count'],
            negative_count=analysis_result['negative_count'],
            product_name=cleaned_product_name,
            save_path=pie_chart_path
        )
        
        # Exclude product, brand and morphs of those from word cloud / bar chart
        exception_list = [brand_name, product_name]
        try:
            exception_list += self.model.mecab.nouns(brand_name) + self.model.mecab.nouns(product_name)
        except Exception:
            pass
        exception_list = list(set(exception_list))
        
        # Plot keywords
        pos_keywords_path = os.path.join(product_dir, f"{cleaned_product_name}_positive_keywords.png")
        self.visualizer.plot_top_keywords(
            noun_lists=analysis_result['positive_nouns'],
            title='Positive Reviews',
            color='#4CAF50',
            exclude_words=exception_list,
            stopwords=self.model.STOPWORDS,
            save_path=pos_keywords_path
        )
        
        neg_keywords_path = os.path.join(product_dir, f"{cleaned_product_name}_negative_keywords.png")
        self.visualizer.plot_top_keywords(
            noun_lists=analysis_result['negative_nouns'],
            title='Negative Reviews',
            color='#F44336',
            exclude_words=exception_list,
            stopwords=self.model.STOPWORDS,
            save_path=neg_keywords_path
        )
        
        print("[+] Visualization reports saved successfully.")
        
        return {
            'product_name': product_name,
            'brand': brand_name,
            'saved_dir': product_dir,
            'final_csv': final_csv_path,
            'sentiment_ratio_chart': pie_chart_path,
            'positive_keywords_chart': pos_keywords_path,
            'negative_keywords_chart': neg_keywords_path,
            'analysis_result': analysis_result
        }
