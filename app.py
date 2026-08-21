"""Gradio API server for the review sentiment analysis live demo.

GRU 감성분석 모델을 REST API로 노출한다. 커스텀 프론트엔드
(demo-gateway/review-analytics/)가 이 API를 호출해 결과를 시각화하며,
Gradio 기본 UI로도 간단히 테스트할 수 있다.

로컬 실행: python app.py  (사전에 python main.py train 으로 모델 학습 필요)
"""
import os
import sys
import json
import math
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import gradio as gr

try:
    # ZeroGPU Space는 @spaces.GPU 함수가 최소 1개 있어야 기동됨.
    # 실제 추론은 CPU로 충분하므로 (방문자별 ZeroGPU 쿼터를 소비하지 않도록)
    # 기동 요건만 더미 함수로 충족하고 모든 처리는 CPU에서 실행한다.
    import spaces

    @spaces.GPU
    def _zerogpu_startup_requirement():
        return None
except ImportError:  # 로컬 실행 환경
    pass

from src.analyzer import SentimentModel

MAX_REVIEWS = 200          # 데모 서버 보호: 한 번에 분석할 최대 리뷰 수
TOP_KEYWORDS = 10
NEUTRAL_LOW, NEUTRAL_HIGH = 0.4, 0.6   # 이 구간은 '판단 유보(중립)'로 분류

# 마침표 없이 이어지는 한국어 리뷰 대응: 종결부호 뒤 + ㅠ/ㅜ 뒤 공백에서 분리
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…~])\s+|(?<=[ㅠㅜ])\s+(?=[가-힣A-Za-z0-9])")


def _split_sentences(text: str):
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
    return parts or [text]

model = SentimentModel(model_dir=os.path.join(BASE_DIR, "models"))
if not model.load():
    raise RuntimeError("models/best_model.h5 + vocab.json 이 필요합니다. `python main.py train` 을 먼저 실행하세요.")

with open(os.path.join(BASE_DIR, "assets", "demo_samples.json"), encoding="utf-8") as f:
    DEMO_SAMPLES = json.load(f)


# 양쪽 감성에 고르게 등장해 판별력이 없는 일반 단어
GENERIC_WORDS = {"구매", "제품", "상품", "물건", "사용", "생각", "주문", "가격", "배송", "그것", "이것", "정도", "때문",
                 "같다", "있다", "진짜", "처음", "부분", "그렇다", "이렇다", "아니다"}


def _discriminative_keywords(pos_docs, neg_docs):
    """
    단순 빈도가 아니라 '반대 감성 대비 얼마나 치우쳐 나오는가'(log-odds)로
    키워드를 뽑는다. 양쪽에 다 나오는 일반 단어는 자동으로 탈락하고,
    각 감성의 '이유'가 되는 단어가 상위로 올라온다.
    문서 빈도(감성 판정된 문장당 1회)를 사용해 반복 단어가 왜곡하지 않게 한다.
    """
    df_pos, df_neg = Counter(), Counter()
    for doc in pos_docs:
        df_pos.update(set(doc))
    for doc in neg_docs:
        df_neg.update(set(doc))

    n_pos, n_neg = max(len(pos_docs), 1), max(len(neg_docs), 1)
    min_df_pos = 2 if len(pos_docs) >= 20 else 1
    min_df_neg = 2 if len(neg_docs) >= 20 else 1

    pos_scored, neg_scored = [], []
    for w in set(df_pos) | set(df_neg):
        if len(w) < 2 or w in GENERIC_WORDS:
            continue
        p = (df_pos[w] + 0.5) / (n_pos + 1)
        n = (df_neg[w] + 0.5) / (n_neg + 1)
        log_odds = math.log(p / n)
        if log_odds > 0 and df_pos[w] >= min_df_pos:
            pos_scored.append((w, log_odds * math.log1p(df_pos[w]), df_pos[w], df_neg[w]))
        elif log_odds < 0 and df_neg[w] >= min_df_neg:
            neg_scored.append((w, -log_odds * math.log1p(df_neg[w]), df_neg[w], df_pos[w]))

    def top(scored):
        scored.sort(key=lambda x: -x[1])
        return [{"word": w, "count": own, "other_count": other} for w, _, own, other in scored[:TOP_KEYWORDS]]

    return top(pos_scored), top(neg_scored)


def analyze(reviews_text: str):
    """줄바꿈으로 구분된 리뷰들을 감성 분석해 JSON으로 반환.

    리뷰 단위 판정과 별개로, 키워드는 '문장' 단위 감성으로 추출한다.
    혼합 감성 리뷰(예: "작아서 아쉬워요 ... 그래도 옷은 예뻐요")에서
    긍정으로 뭉뚱그려져 부정 근거 단어가 사라지는 문제를 막기 위함이다.
    """
    reviews = [r.strip() for r in (reviews_text or "").splitlines() if r.strip()]
    if not reviews:
        raise gr.Error("분석할 리뷰를 한 줄에 하나씩 입력해주세요.")
    reviews = reviews[:MAX_REVIEWS]

    review_scores = model.predict_scores(reviews)

    # 문장 분리 후 전체 문장을 한 번에 배치 추론
    sent_lists = [_split_sentences(r) for r in reviews]
    flat_sentences = [s for sents in sent_lists for s in sents]
    flat_scores = model.predict_scores(flat_sentences)

    items = []
    pos_docs, neg_docs = [], []
    cursor = 0
    for r, s, sents in zip(reviews, review_scores, sent_lists):
        sent_scores = flat_scores[cursor:cursor + len(sents)]
        cursor += len(sents)

        # 키워드 문서는 문장 감성 기준으로 적립 (중립 문장은 어느 쪽에도 기여하지 않음)
        review_words = []
        sentence_items = []
        for sent, ss in zip(sents, sent_scores):
            words = model.tokenizer_ko.content_words(sent)
            review_words.extend(words)
            sentence_items.append({"text": sent, "score": round(ss, 4)})
            if ss > NEUTRAL_HIGH:
                pos_docs.append(words)
            elif ss < NEUTRAL_LOW:
                neg_docs.append(words)

        if s > NEUTRAL_HIGH:
            label = "positive"
        elif s < NEUTRAL_LOW:
            label = "negative"
        else:
            label = "neutral"
        items.append({
            "review": r,
            "score": round(s, 4),                      # 긍정 확률 (0~1)
            "label": label,
            "confidence": round(max(s, 1 - s) * 100, 1),
            "words": sorted(set(review_words)),        # 키워드 매칭용 내용어
            "sentences": sentence_items,               # 문장별 감성 (혼합 리뷰 시각화용)
        })

    pos_keywords, neg_keywords = _discriminative_keywords(pos_docs, neg_docs)
    pos_count = sum(1 for i in items if i["label"] == "positive")
    neg_count = sum(1 for i in items if i["label"] == "negative")
    return {
        "total": len(items),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": len(items) - pos_count - neg_count,
        "positive_ratio": round(pos_count / len(items) * 100, 1),
        "positive_keywords": pos_keywords,
        "negative_keywords": neg_keywords,
        "items": items,
    }


def get_samples():
    """데모용 샘플 리뷰 세트 목록을 반환 (이름/설명/리뷰 텍스트)."""
    return DEMO_SAMPLES


with gr.Blocks(title="Review Analytics API") as demo:
    gr.Markdown(
        """
        # 🛍️ Review Analytics — GRU 감성분석 API
        네이버 쇼핑 리뷰 20만 건으로 학습한 GRU 모델이 리뷰의 긍정/부정을 분류합니다.

        ✨ **커스텀 데모 페이지**: [demo-gateway.trealight112.workers.dev/review-analytics](https://demo-gateway.trealight112.workers.dev/review-analytics/) ·
        📎 [GitHub](https://github.com/Kim-jin-gwang/Shopping-mall-Purchase-Review-Analytics)
        """
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(
                lines=8,
                label="리뷰 입력 (한 줄에 하나씩)",
                placeholder="배송도 빠르고 옷 질도 너무 좋아요!\n사진이랑 색이 달라요. 실망했습니다.",
            )
            btn = gr.Button("감성 분석", variant="primary")
        out = gr.JSON(label="분석 결과")
    btn.click(analyze, inputs=inp, outputs=out, api_name="analyze")

    # 커스텀 FE가 샘플 세트를 가져갈 수 있도록 API로 노출
    sample_out = gr.JSON(visible=False)
    gr.Button("샘플 세트 조회", visible=False).click(get_samples, outputs=sample_out, api_name="samples")

if __name__ == "__main__":
    demo.launch()
