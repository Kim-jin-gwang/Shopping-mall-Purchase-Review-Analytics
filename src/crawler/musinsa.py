import os
import re
import time
import requests
import urllib.request
from typing import Tuple, List, Dict, Any
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

from .base import BaseCrawler

class MusinsaCrawler(BaseCrawler):
    """
    Musinsa review crawler implementing the BaseCrawler interface.
    """
    
    def __init__(self, headless: bool = True):
        self.base_url = "https://www.musinsa.com/"
        self.headless = headless
        
    def _init_browser(self) -> webdriver.Chrome:
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
        
        # Avoid detection issues
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36")
        
        browser = webdriver.Chrome(options=options)
        return browser

    def search_products(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Searches for products in Musinsa and extracts search results using URL patterns.
        """
        browser = self._init_browser()
        products = []
        
        try:
            browser.get(self.base_url)
            time.sleep(2)
            
            # Locate search box and send keys
            search_box = browser.find_element(By.XPATH, '//*[@class="search head-search-inp keyword-dec"]')
            search_box.send_keys(keyword)
            time.sleep(2)
            
            full_html = browser.page_source
            soup = BeautifulSoup(full_html, 'html.parser')
            
            # Find product links matching /goods/\d+
            a_tags = soup.find_all('a', href=re.compile(r'/goods/\d+'))
            
            seen_ids = set()
            for a in a_tags:
                href = a.get('href', '')
                match = re.search(r'/goods/(\d+)', href)
                if match:
                    gid = match.group(1)
                    if gid not in seen_ids:
                        text = a.text.strip()
                        # Filtering out noise elements with short text
                        if len(text) > 5:
                            lines = [line.strip() for line in text.split('\n') if line.strip()]
                            brand = lines[0] if len(lines) > 0 else "Brand"
                            name = lines[1] if len(lines) > 1 else lines[0]
                            
                            products.append({
                                'id': gid,
                                'brand': brand,
                                'name': name
                            })
                            seen_ids.add(gid)
        except Exception as e:
            print(f"[!] Error during searching products: {e}")
        finally:
            browser.quit()
            
        return products

    def crawl_reviews(self, product_id: str, limit: int, has_photo: bool = False) -> Tuple[List[int], List[str]]:
        """
        Crawls review data by hitting Musinsa's internal review API endpoint directly.
        """
        url = "https://www.musinsa.com/api2/review/v1/view/list"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
            "Referer": f"https://www.musinsa.com/goods/{product_id}"
        }
        
        score_list = []
        text_list = []
        page = 1
        count = 0
        has_photo_param = "true" if has_photo else "false"
        
        print(f"[*] Musinsa API crawling started for Product ID: {product_id} (Photo mode: {has_photo_param})")
        
        while count < limit:
            params = {
                "goodsNo": product_id,
                "page": page,
                "pageSize": 20,
                "sort": "up_cnt_desc",
                "myFilter": "false",
                "hasPhoto": has_photo_param
            }
            
            try:
                response = requests.get(url, params=params, headers=headers)
                if response.status_code != 200:
                    print(f"[!] API request failed (HTTP {response.status_code}) on page {page}")
                    break
                    
                res_json = response.json()
                data = res_json.get("data", {})
                review_list = data.get("list", [])
                
                if not review_list:
                    print(f"[+] No more reviews found at page {page}.")
                    break
                    
                for rev in review_list:
                    content = rev.get("content", "").strip()
                    score = rev.get("score", None)
                    
                    # Fallback attributes
                    if not content:
                        content = rev.get("reviewContent", "").strip()
                    if score is None:
                        score = rev.get("point", 0)
                        
                    if content:
                        # Clean whitespace
                        content = content.replace("\t", "").replace("\n", "")
                        text_list.append(content)
                        score_list.append(score)
                        count += 1
                        
                    if count >= limit:
                        break
                        
                print(f"[~] Page {page} fetched: Total collected {count}/{limit} reviews.")
                page += 1
                time.sleep(0.5) # Prevent rate limiting
                
            except Exception as e:
                print(f"[!] Exception occurred during API crawling: {e}")
                break
                
        return score_list, text_list

    def download_product_image(self, product_id: str, output_path: str) -> str:
        """
        Extracts product image URL using OpenGraph tags and downloads it.
        """
        browser = self._init_browser()
        image_saved_path = ""
        
        try:
            browser.get(f"{self.base_url}goods/{product_id}")
            time.sleep(3)
            
            full_html = browser.page_source
            soup = BeautifulSoup(full_html, 'html.parser')
            
            meta_image = soup.find('meta', property='og:image')
            image_src = meta_image['content'] if meta_image else ""
            
            if image_src:
                if image_src.startswith('//'):
                    image_src = 'https:' + image_src
                
                # Make sure target directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Download image
                urllib.request.urlretrieve(image_src, output_path)
                print(f"[+] Product image saved at: {output_path}")
                image_saved_path = output_path
            else:
                print("[!] Meta OpenGraph image tag not found.")
        except Exception as e:
            print(f"[!] Error downloading product image: {e}")
        finally:
            browser.quit()
            
        return image_saved_path
