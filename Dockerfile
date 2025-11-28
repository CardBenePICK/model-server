# 1. Base Image: Use the same Python version as the training environment
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies required by lightgbm
RUN apt-get update && apt-get install -y libgomp1

# 4. Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application files
COPY . .

# 5. Expose the port the app runs on
EXPOSE 9000

# 6. Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
