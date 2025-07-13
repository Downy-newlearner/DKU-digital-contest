#!/usr/bin/env python3
"""
KoELECTRA AI vs Human 텍스트 분류 모델 실행 스크립트
klue/roberta-large 모델 사용 및 엄격한 정제 데이터 적용
"""

import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
import torch.nn.functional as F
import numpy as np
import datetime
import warnings
import sys

# 경고 메시지 필터링
warnings.filterwarnings('ignore')

def print_section(title):
    """섹션 제목을 출력합니다."""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def main():
    print_section("KoELECTRA RoBERTa-Large 텍스트 분류 모델 실행 시작")
    
    # 1. 환경 확인
    print("✅ 라이브러리 로드 완료")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 개수: {torch.cuda.device_count()}")
        print(f"현재 GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. 데이터 파일 확인
    print_section("데이터 파일 확인")
    
    DATA_DIR = "./"
    TRAIN_FILE = "train_strictly_cleaned.csv"  # 엄격한 정제 데이터 사용
    TEST_FILE = "test.csv"
    SUBMISSION_FILE = "sample_submission.csv"
    
    train_file_path = os.path.join(DATA_DIR, TRAIN_FILE)
    test_file_path = os.path.join(DATA_DIR, TEST_FILE)
    submission_file_path = os.path.join(DATA_DIR, SUBMISSION_FILE)
    
    required_files = [train_file_path, test_file_path, submission_file_path]
    for file_path in required_files:
        if not os.path.isfile(file_path):
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            sys.exit(1)
    
    print("✅ 모든 데이터 파일 확인 완료")
    print(f"📊 사용할 훈련 데이터: {TRAIN_FILE}")
    
    # 3. 데이터 로드
    print_section("데이터 로드")
    
    train_data = pd.read_csv(train_file_path, encoding='utf-8')
    test_data = pd.read_csv(test_file_path, encoding='utf-8')
    submission_data = pd.read_csv(submission_file_path, encoding='utf-8')
    
    print(f"Train 데이터: {train_data.shape}")
    print(f"Test 데이터: {test_data.shape}")
    print(f"Submission 데이터: {submission_data.shape}")
    print(f"클래스 분포: {train_data['generated'].value_counts().to_dict()}")
    
    # 4. 모델 설정
    print_section("모델 설정")
    
    MODEL_NAME = "klue/roberta-large"  # RoBERTa-Large 사용
    NUM_CLASSES = 2
    MAX_SEQUENCE_LENGTH = 512
    
    print(f"🤖 모델: {MODEL_NAME}")
    print(f"📏 최대 시퀀스 길이: {MAX_SEQUENCE_LENGTH}")
    
    # 토크나이저 로드
    text_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("✅ 토크나이저 로드 완료")
    
    # 모델 로드
    classification_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES
    )
    print("✅ 모델 로드 완료")
    
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classification_model.to(device)
    print(f"💻 디바이스: {device}")
    
    # 5. 데이터셋 클래스 정의
    class CustomTextDataset(torch.utils.data.Dataset):
        def __init__(self, text_list, label_list=None, max_seq_len=512):
            self.text_encodings = text_tokenizer(
                text_list,
                truncation=True,
                padding=True,
                max_length=max_seq_len,
                return_tensors="pt"
            )
            self.target_labels = label_list

        def __getitem__(self, index):
            data_item = {
                key: tensor[index] for key, tensor in self.text_encodings.items()
            }
            if self.target_labels is not None:
                data_item["labels"] = torch.tensor(self.target_labels[index], dtype=torch.long)
            return data_item

        def __len__(self):
            return len(self.text_encodings["input_ids"])
    
    # 6. 데이터 전처리
    print_section("데이터 전처리")
    
    text_samples = train_data["full_text"].tolist()
    label_samples = train_data["generated"].tolist()
    
    # 훈련/검증 분할
    X_train, X_validation, y_train, y_validation = train_test_split(
        text_samples, label_samples, test_size=0.1, stratify=label_samples, random_state=42
    )
    
    print(f"훈련 데이터: {len(X_train):,}개")
    print(f"검증 데이터: {len(X_validation):,}개")
    
    # 데이터셋 생성
    training_dataset = CustomTextDataset(X_train, y_train, MAX_SEQUENCE_LENGTH)
    validation_dataset = CustomTextDataset(X_validation, y_validation, MAX_SEQUENCE_LENGTH)
    test_dataset = CustomTextDataset(test_data["paragraph_text"].tolist(), max_seq_len=MAX_SEQUENCE_LENGTH)
    
    print("✅ 데이터셋 생성 완료")
    
    # 7. 훈련 설정
    print_section("훈련 설정")
    
    EPOCHS = 5  # RoBERTa-Large는 더 적은 에포크로도 좋은 성능
    BATCH_SIZE = 8  # 메모리 고려하여 배치 크기 감소
    LEARNING_RATE = 1e-5  # RoBERTa-Large는 더 작은 학습률 사용
    OUTPUT_DIR = "./results_roberta_large"
    LOG_DIR = "./logs_roberta_large"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"에포크: {EPOCHS}")
    print(f"배치 크기: {BATCH_SIZE}")
    print(f"학습률: {LEARNING_RATE}")
    print(f"출력 디렉토리: {OUTPUT_DIR}")
    
    # 8. 평가 메트릭 함수
    def calculate_evaluation_metrics(evaluation_predictions):
        predictions, labels = evaluation_predictions
        predictions = np.argmax(predictions, axis=1)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions),
            "precision": precision_score(labels, predictions),
            "recall": recall_score(labels, predictions),
            "roc_auc": roc_auc_score(labels, 
                                   F.softmax(torch.tensor(evaluation_predictions.predictions), dim=1)[:, 1].numpy())
        }
    
    # 9. TrainingArguments 설정
    model_training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        logging_dir=LOG_DIR,
        logging_steps=100,
        save_total_limit=3,
        report_to="none",
        seed=42,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=False,
        warmup_steps=500,
        gradient_accumulation_steps=2
    )
    
    # 10. 훈련 실행
    print_section("모델 훈련")
    
    model_trainer = Trainer(
        model=classification_model,
        args=model_training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=calculate_evaluation_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    start_time = datetime.datetime.now()
    print(f"🚀 훈련 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    training_results = model_trainer.train()
    
    end_time = datetime.datetime.now()
    training_duration = (end_time - start_time).total_seconds()
    
    print(f"✅ 훈련 완료!")
    print(f"훈련 시간: {training_duration // 60:.0f}분 {training_duration % 60:.0f}초")
    
    # 11. 평가 및 예측
    print_section("평가 및 예측")
    
    eval_results = model_trainer.evaluate()
    print("🔍 검증 결과:")
    for key, value in eval_results.items():
        if key.startswith('eval_'):
            print(f"  {key}: {value:.4f}")
    
    # 테스트 예측
    test_predictions = model_trainer.predict(test_dataset)
    raw_probabilities = F.softmax(torch.tensor(test_predictions.predictions), dim=1)[:, 1].numpy()
    
    # 12. 제출 파일 생성
    print_section("제출 파일 생성")
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    auc_score = eval_results.get('eval_roc_auc', 0)
    submission_filename = f"roberta_large_strictly_cleaned_{current_time}_auc_{auc_score:.4f}.csv"
    
    submission_data["generated"] = raw_probabilities
    submission_data.to_csv(submission_filename, index=False)
    
    print(f"✅ 제출 파일 저장: {submission_filename}")
    
    # 13. 모델 저장
    model_save_path = f"./roberta_large_model_{current_time}"
    model_trainer.save_model(model_save_path)
    text_tokenizer.save_pretrained(model_save_path)
    
    print(f"✅ 모델 저장: {model_save_path}")
    
    # 14. 결과 요약
    print_section("실험 결과 요약")
    
    print(f"🤖 모델: {MODEL_NAME}")
    print(f"📊 훈련 데이터: {TRAIN_FILE}")
    print(f"📈 훈련 샘플: {len(X_train):,}개")
    print(f"🔍 검증 샘플: {len(X_validation):,}개")
    print(f"🎯 테스트 샘플: {len(test_data):,}개")
    print(f"⏱️ 훈련 시간: {training_duration // 60:.0f}분 {training_duration % 60:.0f}초")
    print(f"🎉 최종 ROC-AUC: {auc_score:.4f}")
    print(f"📝 제출 파일: {submission_filename}")
    print(f"💾 모델 저장 위치: {model_save_path}")
    
    # 15. 데이터 품질 개선 효과 출력
    print_section("데이터 품질 개선 효과")
    
    print(f"✅ 엄격한 정제 데이터 사용 효과:")
    print(f"   - 문단 수 25개 미만으로 제한")
    print(f"   - 노이즈 데이터 완전 제거")
    print(f"   - 일관된 품질의 텍스트 데이터")
    print(f"   - 모델 학습 안정성 향상")
    
    print_section("실험 완료")
    
    return {
        'model_name': MODEL_NAME,
        'train_file': TRAIN_FILE,
        'auc_score': auc_score,
        'submission_file': submission_filename,
        'model_path': model_save_path,
        'training_time': training_duration
    }

if __name__ == "__main__":
    results = main() 