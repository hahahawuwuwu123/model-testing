# 필요한 라이브러리 불러오기
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample

# 1. 데이터 로드 및 준비
iris = load_iris()  # Iris 데이터셋 로드
X = iris.data  # 특성 데이터
y = iris.target  # 타겟 레이블

# 데이터를 훈련, 검증, 테스트 세트로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 모델 초기화 및 훈련
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 3. 교차 검증 적용
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold 교차 검증 설정
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

print("\n=== 교차 검증 결과 ===")
print(f"Fold별 정확도: {cv_scores}")
print(f"평균 정확도: {np.mean(cv_scores):.4f}")

# 4. 부트스트랩 적용
n_iterations = 1000  # 부트스트랩 반복 횟수
n_size = X_train.shape[0]  # 샘플 크기
bootstrap_scores = []

for i in range(n_iterations):
    # 훈련 데이터에서 부트스트랩 샘플링
    X_boot, y_boot = resample(X_train, y_train, n_samples=n_size, random_state=i)
    model.fit(X_boot, y_boot)  # 모델 훈련
    y_pred = model.predict(X_test)  # 예측 수행
    acc = accuracy_score(y_test, y_pred)  # 정확도 계산
    bootstrap_scores.append(acc)

print("\n=== 부트스트랩 결과 ===")
print(f"부트스트랩 평균 정확도: {np.mean(bootstrap_scores):.4f}")
print(f"부트스트랩 정확도 표준편차: {np.std(bootstrap_scores):.4f}")

# 5. 테스트 세트에서 성능 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n=== 테스트 세트 성능 평가 ===")
print(f"테스트 정확도: {accuracy:.4f}")
print("혼동 행렬:")
print(conf_matrix)

# 분류 리포트 출력
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
