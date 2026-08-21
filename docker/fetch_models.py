"""도커 컨테이너 기동용 모델 부트스트랩.

학습된 GRU 모델(models/best_model.h5, vocab.json)은 용량 문제로 GitHub에는
없고 HF Space 저장소(LFS)에만 있다. 로컬에 모델이 없으면 거기서 받아온다.
(직접 학습하려면: python main.py train)
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SPACE_REPO = "kimjgwang/review-analytics"
FILES = ["models/best_model.h5", "models/vocab.json"]


def main():
    if all(os.path.isfile(os.path.join(BASE_DIR, f)) for f in FILES):
        print("[fetch_models] 모델이 이미 존재합니다 — 다운로드 생략")
        return

    from huggingface_hub import hf_hub_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    for f in FILES:
        print(f"[fetch_models] {SPACE_REPO}(space)에서 {f} 다운로드...")
        hf_hub_download(
            repo_id=SPACE_REPO,
            repo_type="space",
            filename=f,
            local_dir=BASE_DIR,
        )
    print("[fetch_models] 완료")


if __name__ == "__main__":
    main()
