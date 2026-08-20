# 하위 모듈은 필요한 곳에서 직접 import한다.
# (예: from src.analyzer import SentimentModel)
# 패키지 로드 시 크롤러(selenium/bs4) 등 무거운 의존성이 강제로 딸려오지 않도록
# 이 파일에서는 eager import를 하지 않는다.
