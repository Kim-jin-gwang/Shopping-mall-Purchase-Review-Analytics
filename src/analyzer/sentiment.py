import os
import re
import pickle
import urllib.request
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from eunjeon import Mecab

class SentimentModel:
    """
    Handles sentiment classification logic: training, loading, text preprocessing, and inference.
    """
    
    STOPWORDS = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']
    
    def __init__(self, model_dir: str = 'models', max_len: int = 80):
        self.model_dir = model_dir
        self.max_len = max_len
        self.model_path = os.path.join(model_dir, 'best_model.h5')
        self.tokenizer_path = os.path.join(model_dir, 'tokenizer.pickle')
        
        self.model = None
        self.tokenizer = None
        self.mecab = Mecab()
        
    def clean_text(self, text: str) -> str:
        """Removes all characters except Korean alphabet and spaces."""
        if not isinstance(text, str):
            return ""
        return re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", text).strip()
        
    def load(self) -> bool:
        """
        Loads the trained model and tokenizer.
        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path):
            try:
                print(f"[*] Loading model from {self.model_path}...")
                self.model = load_model(self.model_path)
                
                print(f"[*] Loading tokenizer from {self.tokenizer_path}...")
                with open(self.tokenizer_path, 'rb') as handle:
                    self.tokenizer = pickle.load(handle)
                return True
            except Exception as e:
                print(f"[!] Failed to load model/tokenizer: {e}")
                return False
        else:
            print("[!] Model or tokenizer file does not exist.")
            return False
            
    def train(self, dataset_url: str = "https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt") -> None:
        """
        Downloads Naver Shopping review corpus and trains a GRU-based sentiment classifier.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        dataset_path = os.path.join(self.model_dir, 'ratings_total.txt')
        
        # Download dataset if not exists
        if not os.path.exists(dataset_path):
            print(f"[*] Downloading training dataset from {dataset_url}...")
            urllib.request.urlretrieve(dataset_url, filename=dataset_path)
            print("[+] Dataset downloaded successfully.")
            
        print("[*] Loading training dataset...")
        total_data = pd.read_table(dataset_path, names=['ratings', 'reviews'])
        print(f"[*] Total loaded reviews: {len(total_data)}")
        
        # Labeling (rating > 3 -> Positive(1), else Negative(0))
        total_data['label'] = np.select([total_data.ratings > 3], [1], default=0)
        total_data.drop_duplicates(subset=['reviews'], inplace=True)
        print(f"[*] Samples after deduplication: {len(total_data)}")
        
        # Train-Test Split
        train_data, test_data = train_test_split(total_data, test_size=0.25, random_state=42)
        
        print("[*] Preprocessing dataset...")
        train_data['reviews'] = train_data['reviews'].apply(self.clean_text)
        train_data.replace('', np.nan, inplace=True)
        train_data.dropna(subset=['reviews'], inplace=True)
        
        test_data['reviews'] = test_data['reviews'].apply(self.clean_text)
        test_data.replace('', np.nan, inplace=True)
        test_data.dropna(subset=['reviews'], inplace=True)
        
        print("[*] Tokenizing text using Mecab...")
        train_data['tokenized'] = train_data['reviews'].apply(self.mecab.morphs).apply(
            lambda x: [w for w in x if w not in self.STOPWORDS]
        )
        test_data['tokenized'] = test_data['reviews'].apply(self.mecab.morphs).apply(
            lambda x: [w for w in x if w not in self.STOPWORDS]
        )
        
        X_train = train_data['tokenized'].values
        y_train = train_data['label'].values
        X_test = test_data['tokenized'].values
        y_test = test_data['label'].values
        
        print("[*] Fitting Tokenizer...")
        temp_tokenizer = Tokenizer()
        temp_tokenizer.fit_on_texts(X_train)
        
        # Filter rare words
        threshold = 2
        total_cnt = len(temp_tokenizer.word_index)
        rare_cnt = sum(1 for v in temp_tokenizer.word_counts.values() if v < threshold)
        vocab_size = total_cnt - rare_cnt + 2
        
        print(f"[*] Vocabulary size: {total_cnt}, Rare words: {rare_cnt}, Final vocab size: {vocab_size}")
        
        self.tokenizer = Tokenizer(vocab_size, oov_token='OOV')
        self.tokenizer.fit_on_texts(X_train)
        
        X_train_seq = pad_sequences(self.tokenizer.texts_to_sequences(X_train), maxlen=self.max_len)
        X_test_seq = pad_sequences(self.tokenizer.texts_to_sequences(X_test), maxlen=self.max_len)
        
        print("[*] Building GRU model...")
        model = Sequential([
            Embedding(vocab_size, 100),
            GRU(128),
            Dense(1, activation='sigmoid')
        ])
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        # Note: best_model.h5 path is monitored by val_loss/val_acc. Let's use val_loss for checkpoints.
        mc = ModelCheckpoint(self.model_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        
        print("[*] Starting model training...")
        model.fit(X_train_seq, y_train, epochs=5, callbacks=[es, mc], batch_size=128, validation_split=0.2)
        
        self.model = load_model(self.model_path)
        
        # Save Tokenizer
        with open(self.tokenizer_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"[+] Model saved to: {self.model_path}")
        print(f"[+] Tokenizer saved to: {self.tokenizer_path}")
        
        # Evaluation
        loss, accuracy = self.model.evaluate(X_test_seq, y_test, verbose=0)
        print(f"[+] Test Accuracy: {accuracy:.4f}")

    def predict_score(self, text: str) -> float:
        """
        Predicts the sentiment score (positive probability, 0.0 to 1.0) of a single review.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer are not loaded. Call load() or train() first.")
            
        cleaned = self.clean_text(text)
        tokens = self.mecab.morphs(cleaned)
        tokens = [word for word in tokens if word not in self.STOPWORDS]
        
        if not tokens:
            return 0.5
            
        encoded = self.tokenizer.texts_to_sequences([tokens])
        padded = pad_sequences(encoded, maxlen=self.max_len)
        
        score = float(self.model.predict(padded, verbose=0)[0][0])
        return score

    def analyze_reviews(self, reviews: List[str]) -> Dict[str, Any]:
        """
        Analyzes a list of reviews and returns scores, classifications, and tokenized nouns.
        
        Returns:
            Dict containing:
                - 'positive_count': Count of positive reviews
                - 'negative_count': Count of negative reviews
                - 'positive_scores': List of sentiment scores for positive reviews
                - 'negative_scores': List of sentiment scores for negative reviews (1 - positive_score)
                - 'positive_nouns': List of noun lists extracted from positive reviews
                - 'negative_nouns': List of noun lists extracted from negative reviews
        """
        positive_scores = []
        negative_scores = []
        positive_nouns = []
        negative_nouns = []
        
        for r in reviews:
            if not r or not isinstance(r, str):
                continue
                
            score = self.predict_score(r)
            nouns = self.mecab.nouns(r)
            
            if score > 0.5:
                positive_scores.append(score * 100)
                positive_nouns.append(nouns)
            else:
                negative_scores.append((1.0 - score) * 100)
                negative_nouns.append(nouns)
                
        return {
            'positive_count': len(positive_scores),
            'negative_count': len(negative_scores),
            'positive_scores': positive_scores,
            'negative_scores': negative_scores,
            'positive_nouns': positive_nouns,
            'negative_nouns': negative_nouns
        }
