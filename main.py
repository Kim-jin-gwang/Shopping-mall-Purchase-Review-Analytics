import os
import sys
import argparse
import pandas as pd

from src.crawler import MusinsaCrawler
from src.analyzer import SentimentModel
from src.visualization import Visualizer
from src.pipeline import AnalysisPipeline

def run_train(args):
    """
    Trains the sentiment model from scratch.
    """
    print("=========================================")
    print("      Sentiment Model Training Mode      ")
    print("=========================================")
    model = SentimentModel(model_dir=args.model_dir)
    model.train()
    print("[+] Model training and saving complete!")

def run_analyze(args):
    """
    Runs the crawler and sentiment analysis pipeline.
    """
    print("=========================================")
    print("      Integrated Analysis Pipeline       ")
    print("=========================================")
    
    # Initialize components
    crawler = MusinsaCrawler(headless=not args.no_headless)
    model = SentimentModel(model_dir=args.model_dir)
    visualizer = Visualizer()
    
    # Initialize pipeline
    pipeline = AnalysisPipeline(crawler, model, visualizer)
    
    # 1. Search products
    products = pipeline.search(args.keyword)
    if not products:
        print("[!] No products found for the given keyword.")
        return
        
    df_products = pd.DataFrame(products)
    print("\n--- Search Results ---")
    print(df_products[['brand', 'name']])
    
    # 2. Select product
    selected_idx = args.product_index
    if selected_idx is None:
        try:
            val = input(f"\nSelect product number to analyze (0 to {len(products)-1}): ")
            selected_idx = int(val)
        except ValueError:
            print("[!] Invalid input. Please enter a number.")
            return
            
    if selected_idx < 0 or selected_idx >= len(products):
        print(f"[!] Invalid index. Please choose a number between 0 and {len(products)-1}.")
        return
        
    selected_product = products[selected_idx]
    
    # 3. Run Pipeline
    results = pipeline.run(
        keyword=args.keyword,
        selected_product=selected_product,
        limit=args.limit,
        out_dir=args.out_dir
    )
    
    print("\n=========================================")
    print("            Execution Summary            ")
    print("=========================================")
    print(f"Product: [{results['brand']}] {results['product_name']}")
    print(f"Output Directory: {os.path.abspath(results['saved_dir'])}")
    print(f"Combined Review CSV: {os.path.abspath(results['final_csv'])}")
    print(f"Sentiment Ratio Pie Chart: {os.path.abspath(results['sentiment_ratio_chart'])}")
    print(f"Positive Keywords Bar Chart: {os.path.abspath(results['positive_keywords_chart'])}")
    print(f"Negative Keywords Bar Chart: {os.path.abspath(results['negative_keywords_chart'])}")
    print("=========================================")

def main():
    parser = argparse.ArgumentParser(description="Shopping Mall Review Crawler and Sentiment Analytics System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Subparser for 'train'
    train_parser = subparsers.add_parser("train", help="Train the sentiment analysis GRU model")
    train_parser.add_argument("-md", "--model-dir", type=str, default="models", help="Directory to save model/tokenizer")
    
    # Subparser for 'analyze'
    analyze_parser = subparsers.add_parser("analyze", help="Crawl reviews and run sentiment analysis")
    analyze_parser.add_argument("-k", "--keyword", type=str, required=True, help="Search keyword on the shopping mall")
    analyze_parser.add_argument("-l", "--limit", type=int, default=100, help="Number of reviews to crawl (default: 100)")
    analyze_parser.add_argument("-idx", "--product-index", type=int, default=None, help="Index of search results to select (skips interactive input)")
    analyze_parser.add_argument("--no-headless", action="store_true", help="Display Selenium Chrome browser window during execution")
    analyze_parser.add_argument("-o", "--out-dir", type=str, default="data", help="Output directory for crawled data & charts")
    analyze_parser.add_argument("-md", "--model-dir", type=str, default="models", help="Directory where model/tokenizer are stored")
    
    args = parser.parse_args()
    
    if args.command == "train":
        run_train(args)
    elif args.command == "analyze":
        run_analyze(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
