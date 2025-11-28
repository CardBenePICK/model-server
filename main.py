import pandas as pd
import uuid
from typing import List
import traceback

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
    version="1.5"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 모델 로드
try:
    model = load_model('cold_start_automl_champion')
    print("✅ 모델 로드 성공")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")

# 3. Pydantic 스키마 정의
class CustomerProfile(BaseModel):
    AGE: str = Field(..., example="45", description="나이")
    SEX_CD: int = Field(..., example=2, description="성별 (1:남성, 2:여성)")
    LIFE_STAGE: str = Field(..., example="CHILD_UNI", description="라이프 스테이지")
    Q_SPEND: str = Field(..., example="1_Low", description="지출 규모")
    Q_CAR: str = Field(..., example="Yes", description="자차 여부")
    Q_DINING: str = Field(..., example="2_Mid", description="외식 수준")
    Q_LEISURE: str = Field(..., example="No", description="레저 여부")
    Q_EDU: str = Field(..., example="No", description="교육 여부")
    Q_HEALTH: str = Field(..., example="No", description="건강 여부")

class ClusterRanking(BaseModel):
    cluster: int
    probability: float

class PredictionResponse(BaseModel):
    status: str
    request_id: str
    predicted_cluster: int
    confidence_score: float
    ranking: List[ClusterRanking]
    input_check: dict

# 4. 예측 엔드포인트
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_cluster(profile: CustomerProfile):
    try:
        # 1. 고유 ID 생성
        request_id = str(uuid.uuid4())
        
        # 2. 입력 데이터 변환
        input_data = profile.model_dump()
        data_df = pd.DataFrame([input_data])
        
        # 3. 모델 예측 (raw_score=True)
        predictions = predict_model(model, data=data_df, raw_score=True)
        
        # [디버깅 로그] 컬럼 확인용
        print(f"🔍 [DEBUG] Columns: {predictions.columns.tolist()}")

        # 4. 결과 추출 (에러 수정 부분!)
        # 먼저 예측된 라벨(클러스터 번호)을 가져옵니다.
        if 'prediction_label' in predictions.columns:
            predicted_cluster = int(predictions['prediction_label'].iloc[0])
        else:
            predicted_cluster = int(predictions['Label'].iloc[0])
            
        # confidence_score를 동적으로 찾습니다.
        score_col_name = f"prediction_score_{predicted_cluster}" # PyCaret 3.x 표준
        score_col_name_old = f"Score_{predicted_cluster}"       # PyCaret 2.x 호환
        
        if score_col_name in predictions.columns:
            confidence_score = float(predictions[score_col_name].iloc[0])
        elif score_col_name_old in predictions.columns:
            confidence_score = float(predictions[score_col_name_old].iloc[0])
        elif 'prediction_score' in predictions.columns:
            # 만약 단일 컬럼이 살아있다면 그걸 사용
            confidence_score = float(predictions['prediction_score'].iloc[0])
        else:
            # 정말 아무것도 못 찾았을 경우 (예외처리)
            confidence_score = 0.0
            print("⚠️ Confidence Score 컬럼을 찾을 수 없습니다.")

        # 5. 확률 순위 리스트 생성
        ranking_list = []
        scores = {}
        
        for col in predictions.columns:
            # 컬럼명이 'Score_' 또는 'prediction_score_'로 시작하고, 끝이 숫자인 경우
            if (col.startswith("Score_") or col.startswith("prediction_score_")):
                try:
                    parts = col.split('_')
                    # 마지막 부분이 숫자인지 확인 (예: prediction_score_0)
                    if parts[-1].isdigit():
                        cluster_num = int(parts[-1])
                        prob = float(predictions[col].iloc[0])
                        scores[cluster_num] = prob
                except:
                    continue
        
        # 정렬
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for c_num, prob in sorted_scores:
            ranking_list.append(ClusterRanking(cluster=c_num, probability=round(prob, 4)))
            
        return {
            "status": "success",
            "request_id": request_id,
            "predicted_cluster": predicted_cluster,
            "confidence_score": confidence_score,
            "ranking": ranking_list,
            "input_check": input_data
        }

    except Exception as e:
        # 에러 발생 시 상세 로그 출력
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)