# 🛍️ Shopping Mall Purchase Review Analytics (쇼핑몰 구매 리뷰 감성 분석 CLI)

> **Project Date:** 2021.04.25 ~ 2021.05.30  
> **Collaborators:** 양수빈, 장준상  
> **Refactored to CLI & Modular Architecture:** 2026.06.26

기존 별점 시스템의 한계점을 극복하고 사용자의 구매 결정에 실질적인 도움을 주기 위해, 약 200,000개의 쇼핑몰 리뷰 데이터로 사전 학습된 **GRU(Gated Recurrent Unit)** 신경망 모델을 활용해 무신사 상품의 리뷰 데이터를 **감성 분석(Sentiment Analysis)**하여 상품별 최종 **긍정/부정 비율**과 **감정별 대표 키워드 시각화 차트**를 도식화하는 감성 분석 파이프라인 프로젝트입니다.

Jupyter Notebook 환경에서 동작하던 기존 코드베이스를 **유지보수성 및 확장성을 극대화한 모듈식 Python CLI 프로그램**으로 리팩터링하였습니다.

---

## 📂 디렉터리 구조 (Directory Structure)

본 프로젝트는 데이터 수집(크롤러), 모델 학습(GRU), 통합 추론 및 시각화 파이프라인 모듈로 구성되어 있습니다.

```text
Shopping-mall-Purchase-Review-Analytics/
│
├── src/                        # 소스 코드 디렉토리
│   ├── crawler/                # 크롤러 모듈
│   │   ├── base.py             # 크롤러 추상 인터페이스 (BaseCrawler)
│   │   └── musinsa.py          # 무신사 크롤러 구현체 (MusinsaCrawler)
│   │
│   ├── analyzer/               # 감성 분석 및 모델 모듈
│   │   └── sentiment.py        # 감성 예측 및 모델 학습/관리 (SentimentModel)
│   │
│   ├── visualization/          # 시각화 모듈
│   │   └── plot.py             # 차트 시각화 (Visualizer)
│   │
│   └── pipeline.py             # 전체 파이프라인 (AnalysisPipeline)
│
├── main.py                     # CLI 애플리케이션 진입점
├── requirements.txt            # 의존성 목록
├── README.md                   # 프로젝트 상세 가이드
│
# 실행 시 자동 생성되는 디렉토리
├── models/                     # 학습된 모델 및 토크나이저 저장 디렉토리
└── data/                       # 크롤링한 데이터 및 이미지 저장 디렉토리
```

---

## 🚀 설치 및 사용 방법

### **1. 가상환경 구축 및 필수 라이브러리 설치**
의존성 충돌을 방지하기 위해 가상환경을 생성하고 패키지를 설치하는 것을 권장합니다.

```bash
# 가상환경 생성 (Windows/macOS 동일)
python -m venv .venv

# 가상환경 활성화
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 의존성 패키지 설치
pip install -r requirements.txt
```

> ⚠️ **주의사항 (Windows 환경):** 
> 한국어 형태소 분석기인 KoNLPy의 Mecab 모듈을 사용하기 위해 Windows 운영체제에서는 `eunjeon` 패키지가 필수적입니다. (`requirements.txt`에 포함되어 있습니다.)

---

## 💻 CLI 실행 가이드

`main.py` 진입점을 통해 모델 학습 및 크롤링/감성 분석 작업을 수행할 수 있습니다.

### **1. 감성 분석 GRU 모델 학습**
네이버 쇼핑 200,000개 리뷰 데이터를 자동으로 다운로드하여 모델을 처음부터 새로 학습시키고 `models/` 폴더에 체크포인트(`best_model.h5`)와 토크나이저(`tokenizer.pickle`)를 저장합니다.
*(최초 1회 실행 필요, 이미 학습된 파일이 `models/`에 있으면 생략 가능)*

```bash
python main.py train
```
* **옵션:**
  * `-md`, `--model-dir`: 모델과 토크나이저가 저장될 경로를 지정합니다 (기본값: `models`).

---

### **2. 상품 리뷰 크롤링 및 감성 분석 통합 파이프라인 실행**
특정 상품을 무신사에서 검색한 뒤 리뷰를 크롤링하여 감성 분석을 진행하고, 결과 차트를 생성합니다.

```bash
python main.py analyze --keyword "데님" --limit 100
```

* **대화형(Interactive) 실행:**
  위 명령어를 실행하면 검색된 상품 목록이 터미널에 표시되며, 분석을 원하는 상품의 번호를 입력하라는 프롬프트가 나타납니다.
  
* **비대화형/자동화(Non-interactive) 실행:**
  `--product-index` 옵션을 사용하여 특정 상품 번호를 지정함으로써 프롬프트를 생략할 수 있습니다. (배치 작업이나 CI/CD 연동 시 유용)
  ```bash
  python main.py analyze --keyword "데님" --limit 100 --product-index 0
  ```

* **주요 옵션 상세:**
  * `-k`, `--keyword` (필수): 쇼핑몰에서 검색할 키워드.
  * `-l`, `--limit` (선택): 수집할 리뷰 개수 (기본값: `100`).
  * `-idx`, `--product-index` (선택): 검색 결과 중 수집할 상품 인덱스 (0부터 시작).
  * `--no-headless` (선택): 크롤링 진행 시 Selenium Chrome 브라우저 창이 실제로 뜨도록 설정합니다. (디버깅 시 유용)
  * `-o`, `--out-dir` (선택): 크롤링된 CSV 파일과 이미지 차트가 저장될 디렉토리 (기본값: `data`).

---

## 📊 결과 분석 및 아티팩트

분석이 완료되면 설정된 출력 경로(`data/<검색어>/`) 아래에 다음 파일들이 자동으로 생성됩니다:

1. **대표 상품 이미지:** `[상품명].jpg`
2. **리뷰 데이터 CSV:** 
   - 일반 리뷰: `[상품명]_general.csv`
   - 포토/스타일 리뷰: `[상품명]_photo.csv`
   - 전체 병합 리뷰: `[상품명]_final.csv`
3. **감성 비율 파이 차트:** `[상품명]_sentiment_ratio.png`
4. **긍정 리뷰 핵심 키워드 막대 차트:** `[상품명]_positive_keywords.png`
5. **부정 리뷰 핵심 키워드 막대 차트:** `[상품명]_negative_keywords.png`

---

## 🛠️ 확장성 설계 (How to Extend)

본 프로젝트는 다른 쇼핑몰로의 크롤러 이식을 극대화할 수 있도록 설계되었습니다.

### **새로운 쇼핑몰 크롤러 추가하기**
1. `src/crawler/base.py`에 선언된 [BaseCrawler](file:///C:/Users/SSAFY/Desktop/gitgitgit/Shopping-mall-Purchase-Review-Analytics/src/crawler/base.py) 추상 클래스를 상속받습니다.
2. `search_products`, `crawl_reviews`, `download_product_image` 메소드를 각 쇼핑몰의 HTML/API 구조에 맞게 구현합니다.
3. 예시:
```python
# src/crawler/naver_shopping.py
from .base import BaseCrawler

class NaverShoppingCrawler(BaseCrawler):
    def search_products(self, keyword):
        # 네이버 쇼핑 상품 검색 로직 구현
        pass
        
    def crawl_reviews(self, product_id, limit, has_photo=False):
        # 네이버 쇼핑 리뷰 수집 로직 구현
        pass
        
    def download_product_image(self, product_id, output_path):
        # 이미지 다운로드 로직 구현
        pass
```
4. `main.py` 혹은 파이프라인 구동부에서 새로 생성한 크롤러 클래스를 의존성 주입하여 즉시 사용할 수 있습니다.

---

## 🏆 핵심 구현 사항 요약

- **하이브리드 크롤링 도입:** 검색 및 ID 획득은 Selenium, 고속 리뷰 수집은 무신사 내부 REST API 엔드포인트 `/api2/review/v1/view/list` 직접 호출 방식을 채택하여 기존 크롤링 대비 속도 10배 이상 향상.
- **의존성 주입(DI) 패턴:** 파이프라인(`AnalysisPipeline`)이 특정 쇼핑몰 크롤러에 종속되지 않고 `BaseCrawler` 인터페이스에 의존하도록 구현하여 코드의 다형성 확보.
- **데이터 무결성 검증:** 전처리 단계에서 `SettingWithCopyWarning` 및 한글 폰트 깨짐 예외 처리를 완벽하게 차단하여 로버스트한 구동 보장.
