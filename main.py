import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from pycaret.classification import load_model, predict_model
import uvicorn

# 1. FastAPI 앱 초기화
app = FastAPI(
    title="Customer Clustering API",
    description="설문 응답(범주형 데이터)을 기반으로 고객 클러스터를 분류하는 API",
    version="1.2"
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

# 4. 예측 엔드포인트
@app.post("/predict", tags=["Prediction"])
def predict_cluster(profile: CustomerProfile):
    try:
        # 1. 입력 데이터를 딕셔너리로 변환
        input_data = profile.dict()
        
        # 2. 데이터프레임 변환
        # PyCaret 모델은 DataFrame 입력을 기대합니다.
        data_df = pd.DataFrame([input_data])
        
        # 3. 모델 예측
        # PyCaret 파이프라인이 내부적으로 Encoding(One-Hot 등)을 처리합니다.
        predictions = predict_model(model, data=data_df)
        
        # 4. 결과 추출
        # PyCaret 3.x 결과 컬럼: 'prediction_label', 'prediction_score'
        predicted_cluster = predictions['prediction_label'].iloc[0]
        score = predictions['prediction_score'].iloc[0]
        
        # 5. 결과 반환
        return {
            "status": "success",
            "predicted_cluster": int(predicted_cluster),
            "confidence_score": float(score),
            "input_check": input_data # 디버깅용: 입력값 확인
        }

    except Exception as e:
        # 에러 발생 시 상세 메시지 반환
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)