# Review Analytics — GRU 감성분석 Gradio API 서버
# 모델 직렬화 호환을 위해 학습 환경(TF 2.21 / Python 3.12)을 그대로 고정한다.
# 실행: docker build -t review-analytics . && docker run -p 7860:7860 review-analytics
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt huggingface_hub

COPY . .

# 학습된 모델은 GitHub에 없음 — 기동 시 HF Space 저장소(LFS)에서 자동 다운로드
EXPOSE 7860
CMD ["sh", "-c", "python docker/fetch_models.py && python app.py"]
