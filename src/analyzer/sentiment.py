import os
import re
import json
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from kiwipiepy import Kiwi


class KoreanTokenizer:
    """
    Kiwi 기반 한국어 형태소 분석기 래퍼.
    (기존 eunjeon/Mecab은 Windows 및 서버 환경에서 설치가 어려워
    순수 pip 설치가 가능한 kiwipiepy로 교체)
    """

    def __init__(self):
        self.kiwi = Kiwi()

    def morphs(self, text: str) -> List[str]:
        return [t.form for t in self.kiwi.tokenize(text)]

    def nouns(self, text: str) -> List[str]:
        return [t.form for t in self.kiwi.tokenize(text) if t.tag.startswith("NN")]

    # 키워드 분석용 내용어 추출: 명사(NNG/NNP) + 형용사(VA) + 어근(XR).
    # 감성의 '이유'는 명사보다 형용사(좋다/빠르다/늦다/작다)에 담기는 경우가 많다.
    def content_words(self, text: str) -> List[str]:
        words = []
        for t in self.kiwi.tokenize(text):
            if t.tag in ("NNG", "NNP", "XR"):
                words.append(t.form)
            elif t.tag == "VA":
                words.append(t.form + "다")
        return words


class Vocabulary:
    """
    토큰 리스트 <-> 정수 시퀀스 변환기.
    (Keras 3에서 제거된 keras.preprocessing.text.Tokenizer를 대체.
    JSON으로 직렬화되어 파이썬/케라스 버전에 독립적)
    """

    PAD, OOV = 0, 1

    def __init__(self, word_index: Dict[str, int] = None):
        self.word_index = word_index or {}

    @classmethod
    def build(cls, tokenized_texts: List[List[str]], min_count: int = 2, max_words: int = 20000):
        counts: Dict[str, int] = {}
        for tokens in tokenized_texts:
            for w in tokens:
                counts[w] = counts.get(w, 0) + 1
        vocab = [w for w, c in sorted(counts.items(), key=lambda x: -x[1]) if c >= min_count]
        vocab = vocab[: max_words - 2]  # PAD, OOV 자리 확보
        word_index = {w: i + 2 for i, w in enumerate(vocab)}
        return cls(word_index)

    @property
    def size(self) -> int:
        return len(self.word_index) + 2

    def encode(self, tokens: List[str], max_len: int) -> List[int]:
        seq = [self.word_index.get(w, self.OOV) for w in tokens][-max_len:]
        return [self.PAD] * (max_len - len(seq)) + seq  # pre-padding (기존 pad_sequences 기본값과 동일)

    def encode_batch(self, tokenized_texts: List[List[str]], max_len: int) -> np.ndarray:
        return np.array([self.encode(t, max_len) for t in tokenized_texts], dtype=np.int32)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word_index, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


class SentimentModel:
    """
    Handles sentiment classification logic: training, loading, text preprocessing, and inference.
    """

    STOPWORDS = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']

    def __init__(self, model_dir: str = 'models', max_len: int = 80):
        self.model_dir = model_dir
        self.max_len = max_len
        self.model_path = os.path.join(model_dir, 'best_model.h5')
        self.vocab_path = os.path.join(model_dir, 'vocab.json')

        self.model = None
        self.vocab = None
        self.tokenizer_ko = KoreanTokenizer()

    def clean_text(self, text: str) -> str:
        """Removes all characters except Korean alphabet and spaces."""
        if not isinstance(text, str):
            return ""
        return re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", text).strip()

    def tokenize(self, text: str) -> List[str]:
        tokens = self.tokenizer_ko.morphs(self.clean_text(text))
        return [w for w in tokens if w not in self.STOPWORDS]

    def load(self) -> bool:
        """
        Loads the trained model and vocabulary.
        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if os.path.exists(self.model_path) and os.path.exists(self.vocab_path):
            try:
                print(f"[*] Loading model from {self.model_path}...")
                self.model = load_model(self.model_path)

                print(f"[*] Loading vocabulary from {self.vocab_path}...")
                self.vocab = Vocabulary.load(self.vocab_path)
                return True
            except Exception as e:
                print(f"[!] Failed to load model/vocabulary: {e}")
                return False
        else:
            print("[!] Model or vocabulary file does not exist.")
            return False

    def train(self, dataset_url: str = "https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt") -> None:
        """
        Downloads Naver Shopping review corpus and trains a GRU-based sentiment classifier.
        """
        import urllib.request

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

        print("[*] Tokenizing text using Kiwi...")
        train_tokens = [self.tokenize(t) for t in train_data['reviews']]
        train_labels = train_data['label'].values
        test_tokens = [self.tokenize(t) for t in test_data['reviews']]
        test_labels = test_data['label'].values

        # Drop empty samples
        keep = [i for i, t in enumerate(train_tokens) if t]
        train_tokens = [train_tokens[i] for i in keep]
        train_labels = train_labels[keep]
        keep = [i for i, t in enumerate(test_tokens) if t]
        test_tokens = [test_tokens[i] for i in keep]
        test_labels = test_labels[keep]
        print(f"[*] Train samples: {len(train_tokens)}, Test samples: {len(test_tokens)}")

        print("[*] Building vocabulary...")
        self.vocab = Vocabulary.build(train_tokens, min_count=2, max_words=20000)
        self.vocab.save(self.vocab_path)  # 학습 중단에 대비해 미리 저장 (체크포인트와 짝을 이룸)
        print(f"[*] Vocabulary size: {self.vocab.size} (saved to {self.vocab_path})")

        X_train = self.vocab.encode_batch(train_tokens, self.max_len)
        X_test = self.vocab.encode_batch(test_tokens, self.max_len)

        print("[*] Building GRU model...")
        model = Sequential([
            Embedding(self.vocab.size, 100),
            GRU(128),
            Dense(1, activation='sigmoid')
        ])

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        mc = ModelCheckpoint(self.model_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

        print("[*] Starting model training...")
        model.fit(X_train, train_labels, epochs=5, callbacks=[es, mc], batch_size=128, validation_split=0.2)

        self.model = load_model(self.model_path)

        # 배포 파일 크기 절감: 옵티마이저 상태 없이 재저장
        self.model.save(self.model_path, include_optimizer=False)
        self.vocab.save(self.vocab_path)

        print(f"[+] Model saved to: {self.model_path}")
        print(f"[+] Vocabulary saved to: {self.vocab_path}")

        # Evaluation
        loss, accuracy = self.model.evaluate(X_test, test_labels, verbose=0)
        print(f"[+] Test Accuracy: {accuracy:.4f}")

    def predict_score(self, text: str) -> float:
        """
        Predicts the sentiment score (positive probability, 0.0 to 1.0) of a single review.
        """
        if self.model is None or self.vocab is None:
            raise RuntimeError("Model and vocabulary are not loaded. Call load() or train() first.")

        tokens = self.tokenize(text)
        if not tokens:
            return 0.5

        padded = self.vocab.encode_batch([tokens], self.max_len)
        score = float(self.model.predict(padded, verbose=0)[0][0])
        return score

    def predict_scores(self, texts: List[str]) -> List[float]:
        """
        Batch version of predict_score: one model call for many reviews.
        """
        if self.model is None or self.vocab is None:
            raise RuntimeError("Model and vocabulary are not loaded. Call load() or train() first.")

        token_lists = [self.tokenize(t) for t in texts]
        valid_idx = [i for i, t in enumerate(token_lists) if t]
        scores = [0.5] * len(texts)
        if valid_idx:
            padded = self.vocab.encode_batch([token_lists[i] for i in valid_idx], self.max_len)
            preds = self.model.predict(padded, verbose=0).reshape(-1)
            for i, p in zip(valid_idx, preds):
                scores[i] = float(p)
        return scores

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
        reviews = [r for r in reviews if r and isinstance(r, str)]
        scores = self.predict_scores(reviews)

        positive_scores, negative_scores = [], []
        positive_nouns, negative_nouns = [], []

        for r, score in zip(reviews, scores):
            nouns = self.tokenizer_ko.nouns(r)
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
