FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies
# libgomp1: LightGBM 등 ML 라이브러리 필수
# gcc, python3-dev: 일부 Python 패키지 빌드 시 필요할 수 있음 (안정성 확보)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy application files
# (.dockerignore에 .env가 있다면 안전하게 소스코드만 복사됨)
COPY . .

# 6. Expose port
EXPOSE 9000

# 7. Command to run the application
# main.py 안에 uvicorn.run() 코드가 있으므로 python 명령어로 실행해도 되고,
# 아래처럼 uvicorn 명령어로 직접 실행해도 됩니다. (아래 방식 추천)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
