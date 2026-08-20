"""데모용 샘플 리뷰 세트 생성 스크립트.

학습 코퍼스(네이버 쇼핑 리뷰)에서 긍/부정 비율이 다른 3개의 샘플 세트를 추출해
assets/demo_samples.json 으로 저장한다. 라이브 데모에서 '샘플 리뷰 묶음'으로 사용.

사용법: python scripts/prepare_demo_samples.py  (models/ratings_total.txt 필요 — train 시 자동 다운로드됨)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(BASE, "models", "ratings_total.txt")
OUT = os.path.join(BASE, "assets", "demo_samples.json")

# (세트 이름, 설명, 긍정 리뷰 개수, 부정 리뷰 개수)
SETS = [
    ("만족도 높은 상품", "긍정 리뷰가 압도적인 상품의 리뷰 묶음", 102, 18),
    ("호불호 갈리는 상품", "평가가 반반으로 갈리는 상품의 리뷰 묶음", 66, 54),
    ("불만 많은 상품", "부정 리뷰가 많은 상품의 리뷰 묶음", 30, 90),
]


def main():
    if not os.path.exists(CORPUS):
        sys.exit(f"corpus not found: {CORPUS} — run `python main.py train` first")

    df = pd.read_table(CORPUS, names=["ratings", "reviews"])
    df.drop_duplicates(subset=["reviews"], inplace=True)
    df = df[df["reviews"].str.len().between(10, 120)]  # 데모 가독성용 길이 필터
    pos = df[df.ratings > 3]
    neg = df[df.ratings <= 3]

    rng = np.random.RandomState(42)
    samples = []
    for name, desc, n_pos, n_neg in SETS:
        picked = pd.concat([
            pos.sample(n_pos, random_state=rng),
            neg.sample(n_neg, random_state=rng),
        ]).sample(frac=1, random_state=rng)  # shuffle
        samples.append({
            "name": name,
            "description": desc,
            "reviews": picked["reviews"].tolist(),
            "ratings": picked["ratings"].tolist(),
        })
        # 다음 세트와 겹치지 않도록 제거
        pos = pos.drop(picked.index, errors="ignore")
        neg = neg.drop(picked.index, errors="ignore")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)
    print(f"saved {len(samples)} sample sets -> {OUT}")


if __name__ == "__main__":
    main()
