# KoELECTRA 텍스트 분류 모델 (앨리스랩 환경)

## 📋 개요
이 프로젝트는 한국어 ELECTRA 모델을 활용하여 AI가 생성한 텍스트와 사람이 작성한 텍스트를 분류하는 모델을 구현합니다.

## 🎯 목표
- **모델**: monologg/koelectra-base-v3-discriminator
- **작업**: 이진 분류 (Human=0, AI=1)
- **평가지표**: ROC-AUC
- **환경**: 앨리스랩 클라우드

## 📁 프로젝트 구조
```
experiments_koelectra/
├── koelectra_text_classification_alice.ipynb  # 메인 실험 노트북
├── koelectra_text_classification.ipynb       # 원본 Colab 노트북
├── requirements.txt                           # 필요한 라이브러리
├── README.md                                 # 프로젝트 설명서
├── train.csv                                 # 훈련 데이터
├── test.csv                                  # 테스트 데이터
└── sample_submission.csv                     # 제출 형식 샘플
```

## 🚀 실행 방법

### 1. 환경 설정
```bash
# 필요한 라이브러리 설치
pip install -r requirements.txt

# 또는 수동으로 설치
pip install torch transformers scikit-learn pandas numpy
```

### 2. 데이터 확인
다음 파일들이 현재 디렉토리에 있는지 확인하세요:
- `train.csv`: 훈련 데이터 (97,172개 샘플)
- `test.csv`: 테스트 데이터 (1,962개 샘플)
- `sample_submission.csv`: 제출 형식 (1,962개 샘플)

### 3. 노트북 실행
```bash
jupyter notebook koelectra_text_classification_alice.ipynb
```

### 4. 순차 실행
노트북의 모든 셀을 순서대로 실행하세요:
1. 라이브러리 로드 및 환경 확인
2. 데이터 파일 경로 설정
3. 데이터 로드 및 탐색
4. 모델 및 토크나이저 로드
5. 데이터셋 클래스 정의
6. 데이터 전처리 및 분할
7. 훈련 파라미터 설정
8. 평가 메트릭 함수 정의
9. 모델 훈련
10. 평가 및 예측
11. 제출 파일 생성
12. 실험 결과 요약

## ⚙️ 하이퍼파라미터

### 기본 설정
- **모델**: monologg/koelectra-base-v3-discriminator
- **최대 시퀀스 길이**: 512
- **배치 크기**: 16 (앨리스랩 환경 최적화)
- **학습률**: 2e-5
- **에포크**: 10
- **Early Stopping**: 3 epochs patience

### 특수 기법
- **파워 튜닝**: α=1.1 (확률 조정)
- **Mixed Precision**: GPU 사용 시 자동 적용
- **Warmup**: 총 스텝의 10%
- **Weight Decay**: 0.01

## 📊 예상 성능

### 검증 성능 (참고)
- **ROC-AUC**: ~0.970
- **정확도**: ~98%
- **F1 Score**: ~0.87

### 훈련 시간 (추정)
- **GPU 사용**: 약 30-60분
- **CPU 사용**: 약 3-5시간

## 📤 출력 파일

### 제출 파일
- `koelectra_submission_YYYYMMDD_HHMMSS_auc_XXXX.csv`
- 형식: ID, generated (확률값)

### 모델 저장
- `koelectra_model_YYYYMMDD_HHMMSS/`
- 포함: 모델 가중치, 토크나이저, 설정 파일

## 🔧 문제 해결

### 메모리 부족 시
```python
# 배치 크기 조정
BATCH_SIZE = 8  # 기본값 16에서 감소
```

### CUDA 오류 시
```python
# CPU 모드로 전환
device = torch.device("cpu")
```

### 라이브러리 버전 충돌 시
```bash
# 가상 환경 생성
conda create -n koelectra python=3.8
conda activate koelectra
pip install -r requirements.txt
```

## 📋 필요 요구사항

### 시스템 요구사항
- **RAM**: 최소 8GB (권장 16GB)
- **GPU**: NVIDIA GPU (선택사항, 속도 향상)
- **저장공간**: 최소 2GB

### 소프트웨어 요구사항
- **Python**: 3.7+
- **PyTorch**: 1.9+
- **Transformers**: 4.20+
- **Scikit-learn**: 1.0+

## 🎯 성능 개선 방안

### 1. 하이퍼파라미터 튜닝
- 학습률 조정 (1e-5 ~ 5e-5)
- 배치 크기 증가 (GPU 메모리 허용 시)
- 에포크 수 조정

### 2. 모델 변형
- KoELECTRA-large 사용
- 다른 한국어 모델 실험 (KoBERT, KoRoBERTa)

### 3. 데이터 증강
- Back-translation
- 동의어 치환
- 노이즈 추가

### 4. 앙상블
- 여러 모델의 예측 결과 결합
- 가중 평균 또는 투표 방식

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. 모든 필요한 파일이 있는지 확인
2. 라이브러리 버전 호환성 확인
3. GPU 메모리 사용량 확인
4. 로그 메시지 확인

## 📚 참고 자료

- [KoELECTRA 논문](https://arxiv.org/abs/2003.10555)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [2025 SW 중심대학 디지털 경진대회](https://dacon.io/competitions/official/236473/overview/description)

---

**Last Updated**: 2025-01-12  
**Version**: 1.0  
**Environment**: Alice Lab Cloud 