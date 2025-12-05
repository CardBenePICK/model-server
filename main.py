import pandas as pd
import uuid
import os
import json
from typing import List, Optional, Dict
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# PyCaret
from pycaret.classification import load_model, predict_model
import uvicorn

# Database (SQLAlchemy)
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# 0. 환경 변수 및 DB 설정
load_dotenv() 
SQLALCHEMY_DATABASE_URL = os.getenv("DB_URL")

if not SQLALCHEMY_DATABASE_URL:
    print("⚠️ 경고: DB_URL 환경 변수가 설정되지 않았습니다.")
    engine = None
    SessionLocal = None
else:
    try:
        engine = create_engine(SQLALCHEMY_DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        engine = None
        SessionLocal = None

Base = declarative_base()

# ==========================================
# [DB 모델] PredictionLog
# ==========================================
class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(36), unique=True, index=True, comment="UUID")
    
    # 1. 모델 정보 & 입력 데이터
    model_type = Column(String(50), comment="사용 모델 (cold_start / mydata_db)")
    input_data = Column(Text, comment="입력 데이터 전체 (JSON String)")
    
    # 2. 예측 결과
    predicted_cluster = Column(Integer, comment="예측 클러스터")
    confidence_score = Column(Float, comment="확신도")
    
    # 3. 사용자 피드백
    is_correct = Column(Boolean, nullable=True, comment="정답 여부")
    corrected_cluster = Column(Integer, nullable=True, comment="사용자 보정 클러스터")
    comment = Column(Text, nullable=True, comment="코멘트")
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

# 테이블 생성
if engine:
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"⚠️ 테이블 생성 실패: {e}")

def get_db():
    if SessionLocal is None:
        raise HTTPException(status_code=500, detail="Database connection is not configured.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(title="Customer Clustering API", version="3.3", root_path="/model") 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. 모델 로드
# ==========================================
try:
    model_cold = load_model('cold_start_automl_champion')
    print("✅ Cold Start Model Loaded")
except:
    model_cold = None
    print("⚠️ Cold Start Model Load Failed")

try:
    if os.path.exists('mydata_automl_all_stats_champion.pkl'):
        model_mydata = load_model('mydata_automl_all_stats_champion')
        print("✅ MyData Model Loaded")
    else:
        model_mydata = None
        print("⚠️ MyData Model File Not Found")
except:
    model_mydata = None

# ==========================================
# 3. Pydantic 스키마 정의
# ==========================================

class CustomerProfile(BaseModel):
    AGE: str = Field(..., example="45")
    SEX_CD: int = Field(..., example=2)
    LIFE_STAGE: str = Field(..., example="CHILD_UNI")
    Q_SPEND: str = Field(..., example="1_Low")
    Q_CAR: str = Field(..., example="Yes")
    Q_DINING: str = Field(..., example="2_Mid")
    Q_LEISURE: str = Field(..., example="No")
    Q_EDU: str = Field(..., example="No")
    Q_HEALTH: str = Field(..., example="No")

class UserIDRequest(BaseModel):
    user_id: int = Field(..., example=0, description="mock_mydata_storage 테이블의 user_id")

class ClusterRanking(BaseModel):
    cluster: int
    probability: float

class PredictionResponse(BaseModel):
    status: str
    request_id: str
    predicted_cluster: int
    confidence_score: float
    ranking: List[ClusterRanking]

class FeedbackRequest(BaseModel):
    request_id: str
    is_correct: bool
    corrected_cluster: Optional[int] = None
    comment: Optional[str] = None

# 4. Helper Functions
def save_prediction_log(db: Session, request_id: str, model_type: str, input_dict: dict, prediction: int, score: float):
    try:
        input_json = json.dumps(input_dict, ensure_ascii=False)
        
        log_entry = PredictionLog(
            request_id=request_id,
            model_type=model_type,
            input_data=input_json,
            predicted_cluster=prediction,
            confidence_score=score
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        print(f"❌ 로그 저장 실패: {e}")

# ==========================================
# 5. API 엔드포인트
# ==========================================

# 1. Cold Start 예측
@app.post("/predict", response_model=PredictionResponse)
def predict_cluster(profile: CustomerProfile, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if model_cold is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = str(uuid.uuid4())
    input_data = profile.model_dump()
    
    data_df = pd.DataFrame([input_data])
    predictions = predict_model(model_cold, data=data_df, raw_score=True)
    
    # 1. Label 추출
    if 'prediction_label' in predictions.columns:
        pred_label = int(predictions['prediction_label'].iloc[0])
    elif 'Label' in predictions.columns:
        pred_label = int(predictions['Label'].iloc[0])
    else:
        raise HTTPException(status_code=500, detail="Prediction failed: No label found.")
        
    # 2. Score 추출
    if 'prediction_score' in predictions.columns:
        pred_score = float(predictions['prediction_score'].iloc[0])
    elif 'Score' in predictions.columns:
        pred_score = float(predictions['Score'].iloc[0])
    else:
        pred_score = 0.0 # 일단 0.0으로 두고 아래 랭킹 로직에서 채움

    # 3. 랭킹 계산 (로그 저장보다 먼저 수행)
    ranking_list = []
    for col in predictions.columns:
        if (col.startswith("prediction_score_") or col.startswith("Score_")) and col.split('_')[-1].isdigit():
            cluster_num = int(col.split('_')[-1])
            prob = float(predictions[col].iloc[0])
            ranking_list.append(ClusterRanking(cluster=cluster_num, probability=round(prob, 4)))
            
    ranking_list.sort(key=lambda x: x.probability, reverse=True)
    
    # Score가 0.0이면 랭킹 1위 값을 가져옴
    if pred_score == 0.0 and ranking_list:
        pred_score = ranking_list[0].probability

    # 4. 로그 저장
    background_tasks.add_task(save_prediction_log, db, request_id, "cold_start", input_data, pred_label, pred_score)
    
    return {
        "status": "success",
        "request_id": request_id,
        "predicted_cluster": pred_label,
        "confidence_score": pred_score,
        "ranking": ranking_list
    }

# 2. MyData 예측 (DB 직접 조회)
@app.post("/predict/mydata", response_model=PredictionResponse)
def predict_mydata_by_id(
    req: UserIDRequest, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
    if model_mydata is None:
        raise HTTPException(status_code=503, detail="MyData model not loaded")
    
    request_id = str(uuid.uuid4())
    
    # -------------------------------------------------------
    # 1. DB에서 user_id로 데이터 조회
    # -------------------------------------------------------
    try:
        query = f"SELECT * FROM mock_mydata_storage WHERE user_id = {req.user_id}"
        data_df = pd.read_sql(query, engine)
        
        if data_df.empty:
             raise HTTPException(status_code=404, detail=f"User ID {req.user_id} not found in mock storage.")
             
        data_df['AGE'] = data_df['AGE'].astype(str)
        data_df['SEX_CD'] = data_df['SEX_CD'].astype(int)
        
        if 'user_id' in data_df.columns:
            data_df = data_df.drop(columns=['user_id'])
            
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"DB Fetch Error: {e}")
        raise HTTPException(status_code=500, detail="Database error occurred during fetch.")

    # -------------------------------------------------------
    # 2. 예측 수행
    # -------------------------------------------------------
    predictions = predict_model(model_mydata, data=data_df, raw_score=True)
    
    # -------------------------------------------------------
    # 3. 결과 추출 및 랭킹 계산
    # -------------------------------------------------------
    
    # (1) 라벨 추출
    if 'prediction_label' in predictions.columns:
        pred_label = int(predictions['prediction_label'].iloc[0])
    elif 'Label' in predictions.columns:
        pred_label = int(predictions['Label'].iloc[0])
    else:
        raise HTTPException(status_code=500, detail="Model prediction failed: No label found.")

    # (2) 점수 추출
    if 'prediction_score' in predictions.columns:
        pred_score = float(predictions['prediction_score'].iloc[0])
    elif 'Score' in predictions.columns:
        pred_score = float(predictions['Score'].iloc[0])
    else:
        pred_score = 0.0 # 없으면 0.0으로 설정

    # (3) 랭킹 로직 실행
    ranking_list = []
    for col in predictions.columns:
        # prediction_score_0, prediction_score_1 등의 컬럼을 찾아서 리스트화
        if (col.startswith("prediction_score_") or col.startswith("Score_")) and col.split('_')[-1].isdigit():
            cluster_num = int(col.split('_')[-1])
            prob = float(predictions[col].iloc[0])
            ranking_list.append(ClusterRanking(cluster=cluster_num, probability=round(prob, 4)))
            
    ranking_list.sort(key=lambda x: x.probability, reverse=True)
    
    # (4) 점수 보정 (통합 점수 컬럼이 없었을 경우, 랭킹 1위를 점수로 채택)
    if pred_score == 0.0 and ranking_list:
        pred_score = ranking_list[0].probability

    # -------------------------------------------------------
    # 4. 로그 저장 (Background Task) 
    # -------------------------------------------------------
    # 이제 pred_score가 정해졌으므로 로그에 저장
    input_log_data = data_df.iloc[0].to_dict()
    
    background_tasks.add_task(
        save_prediction_log, 
        db, 
        request_id, 
        "mydata_db_lookup", 
        input_log_data, 
        pred_label, 
        pred_score
    )

    return {
        "status": "success",
        "request_id": request_id,
        "predicted_cluster": pred_label,
        "confidence_score": pred_score,
        "ranking": ranking_list
    }

# 3. 피드백 업데이트
@app.post("/feedback")
def update_feedback(feedback: FeedbackRequest, db: Session = Depends(get_db)):
    try:
        log_entry = db.query(PredictionLog).filter(PredictionLog.request_id == feedback.request_id).first()
        
        if not log_entry:
            raise HTTPException(status_code=404, detail="해당 request_id를 가진 로그가 없습니다.")
        
        log_entry.is_correct = feedback.is_correct
        log_entry.corrected_cluster = feedback.corrected_cluster
        log_entry.comment = feedback.comment
        log_entry.updated_at = datetime.now()
        
        db.commit()
        return {"status": "success", "message": "피드백 반영 완료"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)