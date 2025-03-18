 
FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y espeak

RUN pip install --upgrade pip setuptools wheel


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .


COPY zonos/ ./zonos/


CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
