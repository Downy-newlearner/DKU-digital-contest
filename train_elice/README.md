# 🚀 KoELECTRA AI vs Human 텍스트 분류 - 앨리스랩 실행 가이드

## 📋 프로젝트 개요
- **모델**: klue/roberta-large
- **데이터**: 엄격하게 정제된 훈련 데이터 (문단 수 < 25)
- **목표**: AI 생성 텍스트 vs 인간 작성 텍스트 분류
- **예상 성능**: ROC-AUC 0.83+

## 🔧 앨리스랩에서 실행하기

### 1. 앨리스랩 접속 및 준비
1. 앨리스랩에 로그인 후 GPU 인스턴스 생성
2. Jupyter Notebook 또는 Terminal 환경 선택
3. Python 3.8+ 환경 확인

### 2. 파일 업로드
다음 파일들을 작업 디렉토리에 업로드:
```
train_strictly_cleaned.csv  # 엄격하게 정제된 훈련 데이터
test.csv                   # 테스트 데이터
sample_submission.csv      # 제출 샘플 파일
run_experiment_roberta.py  # 실험 실행 스크립트
requirements.txt          # 필요한 라이브러리 목록
```

### 3. 라이브러리 설치
```bash
# 필요한 라이브러리 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install torch transformers scikit-learn pandas numpy
```

### 4. 🚀 모델 훈련 실행
```bash
# 터미널에서 실행
python run_experiment_roberta.py
```

### 5. 📊 실행 과정 모니터링
스크립트 실행 시 다음과 같은 진행 상황을 확인할 수 있습니다:

```
============================================================
🎯 KoELECTRA RoBERTa-Large 텍스트 분류 모델 실행 시작
============================================================

✅ 라이브러리 로드 완료
PyTorch 버전: 2.x.x
CUDA 사용 가능: True
GPU 개수: 1
현재 GPU: Tesla T4 (또는 사용 가능한 GPU)

============================================================
🎯 데이터 파일 확인
============================================================

✅ 모든 데이터 파일 확인 완료
📊 사용할 훈련 데이터: train_strictly_cleaned.csv

... [훈련 진행] ...
```

### 6. 📈 예상 실행 시간 및 성능
- **GPU 환경**: 약 30-60분
- **CPU 환경**: 약 2-4시간 (권장하지 않음)
- **예상 메모리 사용량**: 6-8GB
- **목표 ROC-AUC**: 0.83+

### 7. 🎯 결과 파일 확인
훈련 완료 후 다음 파일들이 생성됩니다:
```
roberta_large_strictly_cleaned_YYYYMMDD_HHMMSS_auc_X.XXXX.csv  # 제출 파일
roberta_large_model_YYYYMMDD_HHMMSS/                          # 저장된 모델
results_roberta_large/                                        # 훈련 결과
logs_roberta_large/                                           # 훈련 로그
```

## 🔥 Cursor AI에게 전달할 명령어

앨리스랩 원격 연결 후 Cursor AI에게 다음과 같이 요청하세요:

```
앨리스랩 GPU 환경에서 다음 작업을 수행해주세요:

1. 현재 디렉토리에 있는 파일들 확인:
   - train_strictly_cleaned.csv (87,620개 샘플)
   - test.csv
   - sample_submission.csv  
   - run_experiment_roberta.py
   - requirements.txt

2. 라이브러리 설치:
   pip install -r requirements.txt

3. 모델 훈련 실행:
   python run_experiment_roberta.py

4. 훈련 완료 후 제출 파일 확인 및 성능 보고

모델: klue/roberta-large
데이터: 엄격한 정제 데이터 (문단 수 < 25개)
목표: ROC-AUC 0.83+ 달성
```

## ⚠️ 주의사항

### GPU 메모리 부족 시:
`run_experiment_roberta.py` 파일에서 다음 값들을 조정:
```python
BATCH_SIZE = 4  # 8에서 4로 감소
gradient_accumulation_steps = 4  # 2에서 4로 증가
```

### 훈련 시간 단축하려면:
```python
EPOCHS = 3  # 5에서 3으로 감소
```

### 더 엄격한 Early Stopping:
```python
EarlyStoppingCallback(early_stopping_patience=1)  # 2에서 1로 감소
```

## 📞 문제 해결

1. **CUDA 오류**: GPU 드라이버 재시작 또는 인스턴스 재시작
2. **메모리 부족**: 배치 크기 감소
3. **모델 다운로드 실패**: 인터넷 연결 확인
4. **권한 오류**: 파일 권한 확인 (`chmod +x run_experiment_roberta.py`)

## 🎉 성공 기준
- 훈련 완료 메시지 출력
- ROC-AUC 0.80+ 달성
- 제출 파일 정상 생성

---
**Happy Training! 🚀** 