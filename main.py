import pandas as pd
import uuid
import os
from typing import List, Optional
import traceback
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# PyCaret
from pycaret.classification import load_model, predict_model
import uvicorn

# Database (SQLAlchemy)
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session



# 0. 환경 변수 및 DB 설정
# .env 파일 로드
load_dotenv() 

# 환경 변수에서 DB_URL 가져오기
SQLALCHEMY_DATABASE_URL = os.getenv("DB_URL")

# 예외 처리: .env 파일이 없거나 설정이 안 되어 있을 경우를 대비
if not SQLALCHEMY_DATABASE_URL:
    raise ValueError("❌ DB_URL 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# DB 모델 정의
class Feedback(Base):
    __tablename__ = "customer_feedback"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(36), index=True, comment="예측 요청 시 발급된 UUID")
    
    # 예측 당시 정보 (나중에 분석을 위해 저장)
    predicted_cluster = Column(Integer, comment="모델이 예측한 클러스터")
    confidence_score = Column(Float, comment="모델의 확신도")
    
    # 피드백 정보
    is_correct = Column(Boolean, comment="예측이 맞는지 여부")
    corrected_cluster = Column(Integer, nullable=True, comment="사용자가 생각하는 실제 클러스터 (틀렸을 경우)")
    comment = Column(Text, nullable=True, comment="사용자 추가 코멘트")
    
    created_at = Column(DateTime, default=datetime.now)

# 테이블 생성 (서버 시작 시 테이블이 없으면 생성)
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"⚠️ 데이터베이스 연결 실패 (로컬 테스트가 아니라면 확인 필요): {e}")

# DB 세션 의존성 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 1. FastAPI 앱 초기화
# ==========================================
app = FastAPI(
    title="Customer Clustering API + Feedback",
    description="고객 클러스터 분류 및 사용자 피드백 수집 API",
    version="2.0"
)

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
    model = load_model('cold_start_automl_champion')
    print("✅ 모델 로드 성공")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")

# ==========================================
# 3. Pydantic 스키마 정의
# ==========================================
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

# [신규] 피드백 요청 스키마
class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="predict API에서 반환받은 request_id")
    predicted_cluster: int = Field(..., description="모델이 예측했던 클러스터 번호")
    confidence_score: float = Field(..., description="모델의 확신도 (선택 저장용)")
    is_correct: bool = Field(..., description="예측이 맞으면 True, 틀리면 False")
    corrected_cluster: Optional[int] = Field(None, description="틀렸다면 실제 클러스터 번호")
    comment: Optional[str] = Field(None, description="추가 의견")

# ==========================================
# 4. API 엔드포인트
# ==========================================

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
        
        # 4. 결과 추출 로직
        if 'prediction_label' in predictions.columns:
            predicted_cluster = int(predictions['prediction_label'].iloc[0])
        else:
            predicted_cluster = int(predictions['Label'].iloc[0])
            
        score_col_name = f"prediction_score_{predicted_cluster}"
        score_col_name_old = f"Score_{predicted_cluster}"
        
        if score_col_name in predictions.columns:
            confidence_score = float(predictions[score_col_name].iloc[0])
        elif score_col_name_old in predictions.columns:
            confidence_score = float(predictions[score_col_name_old].iloc[0])
        elif 'prediction_score' in predictions.columns:
            confidence_score = float(predictions['prediction_score'].iloc[0])
        else:
            confidence_score = 0.0

        # 5. 확률 순위 리스트 생성
        ranking_list = []
        scores = {}
        
        for col in predictions.columns:
            if (col.startswith("Score_") or col.startswith("prediction_score_")):
                try:
                    parts = col.split('_')
                    if parts[-1].isdigit():
                        cluster_num = int(parts[-1])
                        prob = float(predictions[col].iloc[0])
                        scores[cluster_num] = prob
                except:
                    continue
        
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback", tags=["Feedback"])
def save_feedback(feedback: FeedbackRequest, db: Session = Depends(get_db)):
    """
    사용자로부터 예측 결과에 대한 피드백(정답 여부, 실제 정답)을 받아 DB에 저장합니다.
    """
    try:
        # DB 모델 객체 생성
        db_feedback = Feedback(
            request_id=feedback.request_id,
            predicted_cluster=feedback.predicted_cluster,
            confidence_score=feedback.confidence_score,
            is_correct=feedback.is_correct,
            corrected_cluster=feedback.corrected_cluster,
            comment=feedback.comment
        )
        
        # 저장 수행
        db.add(db_feedback)
        db.commit()
        db.refresh(db_feedback)
        
        return {
            "status": "success",
            "message": "피드백이 성공적으로 저장되었습니다.",
            "feedback_id": db_feedback.id
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"DB 저장 중 오류 발생: {str(e)}")

@app.get("/health", tags=["Operations"])
def health_check(db: Session = Depends(get_db)):
    """
    서버 상태 및 DB 연결 상태를 점검합니다. (로드밸런서/K8s용)
    """
    try:
        # DB에 간단한 쿼리 실행 (SELECT 1)
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected", "timestamp": datetime.now()}
    except Exception as e:
        return {"status": "unhealthy", "database": str(e), "timestamp": datetime.now()}
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)