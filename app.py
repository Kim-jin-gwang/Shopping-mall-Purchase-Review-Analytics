"""Gradio API server for the review sentiment analysis live demo.

GRU 감성분석 모델을 REST API로 노출한다. 커스텀 프론트엔드
(demo-gateway/review-analytics/)가 이 API를 호출해 결과를 시각화하며,
Gradio 기본 UI로도 간단히 테스트할 수 있다.

로컬 실행: python app.py  (사전에 python main.py train 으로 모델 학습 필요)
"""
import os
import sys
import json
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

model = SentimentModel(model_dir=os.path.join(BASE_DIR, "models"))
if not model.load():
    raise RuntimeError("models/best_model.h5 + vocab.json 이 필요합니다. `python main.py train` 을 먼저 실행하세요.")

with open(os.path.join(BASE_DIR, "assets", "demo_samples.json"), encoding="utf-8") as f:
    DEMO_SAMPLES = json.load(f)


def _top_keywords(noun_lists, exclude=()):
    counter = Counter()
    for nouns in noun_lists:
        counter.update(n for n in nouns if len(n) > 1 and n not in exclude)
    return [{"word": w, "count": c} for w, c in counter.most_common(TOP_KEYWORDS)]


def analyze(reviews_text: str):
    """줄바꿈으로 구분된 리뷰들을 감성 분석해 JSON으로 반환."""
    reviews = [r.strip() for r in (reviews_text or "").splitlines() if r.strip()]
    if not reviews:
        raise gr.Error("분석할 리뷰를 한 줄에 하나씩 입력해주세요.")
    reviews = reviews[:MAX_REVIEWS]

    scores = model.predict_scores(reviews)
    items = []
    pos_nouns, neg_nouns = [], []
    for r, s in zip(reviews, scores):
        positive = s > 0.5
        items.append({
            "review": r,
            "score": round(s, 4),                      # 긍정 확률 (0~1)
            "label": "positive" if positive else "negative",
            "confidence": round((s if positive else 1 - s) * 100, 1),
        })
        nouns = model.tokenizer_ko.nouns(r)
        (pos_nouns if positive else neg_nouns).append(nouns)

    pos_count = sum(1 for i in items if i["label"] == "positive")
    return {
        "total": len(items),
        "positive_count": pos_count,
        "negative_count": len(items) - pos_count,
        "positive_ratio": round(pos_count / len(items) * 100, 1),
        "positive_keywords": _top_keywords(pos_nouns),
        "negative_keywords": _top_keywords(neg_nouns),
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
