# 🚀 빠른 실행 가이드 (앨리스랩 환경)

## 1분 빠른 실행

### 1. 환경 설정
```bash
# 라이브러리 설치
pip install -r requirements.txt
```

### 2. 실행 방법 (2가지 옵션)

#### 옵션 A: 노트북 실행 (권장)
```bash
jupyter notebook koelectra_text_classification_alice.ipynb
```

#### 옵션 B: 스크립트 실행
```bash
python run_experiment.py
```

### 3. 예상 실행 시간
- **GPU 사용**: 약 30-60분
- **CPU 사용**: 약 3-5시간

### 4. 출력 파일
- `koelectra_submission_YYYYMMDD_HHMMSS_auc_XXXX.csv` (제출 파일)
- `koelectra_model_YYYYMMDD_HHMMSS/` (모델 저장)

## 📋 필요 파일 확인

현재 디렉토리에 다음 파일들이 있는지 확인하세요:

✅ `train.csv` (512MB)  
✅ `test.csv` (1.4MB)  
✅ `sample_submission.csv` (23KB)  
✅ `requirements.txt`  
✅ `koelectra_text_classification_alice.ipynb`  
✅ `run_experiment.py`  

## 🔧 문제 해결

### 메모리 부족 시
```python
# 배치 크기 조정
BATCH_SIZE = 8  # 기본값 16에서 감소
```

### GPU 오류 시
```python
# CPU 모드로 전환
device = torch.device("cpu")
```

---

💡 **자세한 설명은 `README.md`를 참조하세요.** 