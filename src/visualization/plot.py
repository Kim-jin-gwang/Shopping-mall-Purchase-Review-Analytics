import os
from collections import Counter
from typing import List, Optional
import matplotlib.pyplot as plt

class Visualizer:
    """
    Handles plotting sentiment ratio and keyword frequency charts.
    """
    
    def __init__(self):
        self._setup_korean_font()
        
    def _setup_korean_font(self):
        """Set up matplotlib font to support Korean characters."""
        try:
            from matplotlib import font_manager, rc
            if os.name == 'nt': # Windows
                font_path = "C:/Windows/Fonts/malgun.ttf"
                font_name = font_manager.FontProperties(fname=font_path).get_name()
                rc('font', family=font_name)
            else: # Mac/Linux fallback
                rc('font', family='AppleGothic')
        except Exception as e:
            print(f"[!] Warning: Matplotlib Korean font setup failed: {e}")

    def plot_sentiment_ratio(self, positive_count: int, negative_count: int, product_name: str, save_path: Optional[str] = None) -> None:
        """
        Draws and optionally saves a pie chart showing the positive/negative review ratio.
        """
        total = positive_count + negative_count
        if total == 0:
            print("[!] No sentiment data to plot ratio.")
            return
            
        ratios = [(positive_count / total) * 100, (negative_count / total) * 100]
        labels = ['Positive', 'Negative']
        colors = ['#4CAF50', '#F44336']
        
        plt.figure(figsize=(6, 6))
        plt.pie(ratios, labels=labels, autopct='%.2f%%', colors=colors, startangle=90, counterclock=False)
        plt.legend(labels)
        plt.title(f'<{product_name}>\nReview Sentiment Ratio')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"[+] Sentiment ratio pie chart saved at: {save_path}")
            
        plt.close()

    def plot_top_keywords(self, noun_lists: List[List[str]], title: str, color: str, 
                          exclude_words: Optional[List[str]] = None, stopwords: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> None:
        """
        Draws and optionally saves a bar chart displaying top keywords.
        """
        # Flatten the list of lists of nouns
        all_nouns = [noun for nouns in noun_lists for noun in nouns]
        
        # Build filter lists
        exclude = set(exclude_words) if exclude_words else set()
        stop = set(stopwords) if stopwords else set()
        
        # Filter nouns (length > 1, not in exclude/stop)
        filtered_nouns = [
            n for n in all_nouns 
            if len(n) > 1 and n not in exclude and n not in stop
        ]
        
        counter = Counter(filtered_nouns)
        top_keywords = counter.most_common(15)
        
        if not top_keywords:
            print(f"[!] Insufficient keywords to plot for '{title}'")
            return
            
        words = [item[0] for item in top_keywords]
        counts = [item[1] for item in top_keywords]
        
        plt.figure(figsize=(10, 5))
        plt.bar(words, counts, color=color)
        plt.title(f'{title} - Top Keywords')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Counts')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"[+] Top keywords bar chart saved at: {save_path}")
            
        plt.close()
