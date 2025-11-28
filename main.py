import pandas as pd
import uuid
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# PyCaret
from pycaret.classification import load_model, predict_model
import uvicorn

# 1. FastAPI 앱 초기화
app = FastAPI(
    title="Customer Clustering API",
    description="설문 응답(범주형 데이터)을 기반으로 고객 클러스터를 분류하는 API (확률 순위 포함)",
    version="1.3"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 허용할 프론트엔드 주소들
    allow_credentials=True,     # 쿠키/인증정보 허용 여부
    allow_methods=["*"],        # 허용할 HTTP Method (GET, POST, OPTIONS 등 전체 허용)
    allow_headers=["*"],        # 허용할 헤더 (Content-Type 등 전체 허용)
)

# 2. 모델 로드
# 모델 파일(cold_start_automl_champion.pkl)이 같은 폴더에 있어야 합니다.
model = load_model('cold_start_automl_champion')

# 3. 입력 데이터 스키마 정의 (설문 응답형)
# 프론트엔드에서 선택된 값("1_Low", "Yes" 등)을 그대로 받습니다.
class CustomerProfile(BaseModel):
    # 인구통계 정보
    AGE: str = Field(..., example="45", description="나이 (실제 데이터 포맷에 맞춤, 예: '45')")
    SEX_CD: int = Field(..., example=2, description="성별 코드 (1: 남성, 2: 여성)")
    LIFE_STAGE: str = Field(..., example="CHILD_UNI", description="라이프 스테이지 (예: CHILD_UNI, NEW_JOB)")
    
    # 설문 응답 정보 (범주형)
    Q_SPEND: str = Field(..., example="1_Low", description="지출 규모 (1_Low, 2_Mid, 3_High)")
    Q_CAR: str = Field(..., example="Yes", description="자차 보유/소비 여부 (Yes, No)")
    Q_DINING: str = Field(..., example="2_Mid", description="외식 소비 수준 (1_Low, 2_Mid, 3_High)")
    Q_LEISURE: str = Field(..., example="No", description="여행/레저 소비 여부 (Yes, No)")
    Q_EDU: str = Field(..., example="No", description="교육 소비 여부 (Yes, No)")
    Q_HEALTH: str = Field(..., example="No", description="건강/병원 소비 여부 (Yes, No)")

# 순위 정보 스키마
class ClusterRanking(BaseModel):
    cluster: int
    probability: float

# 최종 응답 스키마
class PredictionResponse(BaseModel):
    status: str
    request_id: str
    predicted_cluster: int
    confidence_score: float
    ranking: List[ClusterRanking] # 1위부터 순서대로 담길 리스트
    input_check: dict


# 4. 예측 엔드포인트
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_cluster(profile: CustomerProfile):
    try:
        # 1. 고유 ID 생성 (요청 추적용)
        request_id = str(uuid.uuid4())
        
        # 2. 입력 데이터 변환
        input_data = profile.model_dump()
        data_df = pd.DataFrame([input_data])
        
        # 3. 모델 예측 (raw_score=True 옵션 추가)
        predictions = predict_model(model, data=data_df, raw_score=True)
        
        # 4. 결과 추출 (1순위 예측값)
        predicted_cluster = int(predictions['prediction_label'].iloc[0])
        confidence_score = float(predictions['prediction_score'].iloc[0])
        
        # 5. [핵심] 확률 순위 리스트 생성 로직
        ranking_list = []
        scores = {}
        
        # 데이터프레임 컬럼 중 'Score_'로 시작하는 것만 찾습니다.
        for col in predictions.columns:
            if col.startswith("Score_"):
                try:
                    # 컬럼명 예시: Score_0 -> 0번 클러스터
                    cluster_num = int(col.split('_')[1])
                    prob = float(predictions[col].iloc[0])
                    scores[cluster_num] = prob
                except:
                    continue
        
        # 확률이 높은 순서대로 정렬 (내림차순)
        # sorted_scores는 [(클러스터번호, 확률), ...] 형태의 리스트가 됨
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 응답 포맷에 맞게 변환
        for c_num, prob in sorted_scores:
            ranking_list.append(ClusterRanking(cluster=c_num, probability=round(prob, 4)))
            
        # 6. 결과 반환
        return {
            "status": "success",
            "request_id": request_id,
            "predicted_cluster": predicted_cluster,
            "confidence_score": confidence_score,
            "ranking": ranking_list,
            "input_check": input_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)