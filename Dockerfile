FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


COPY runs/ /app/runs/
COPY classification.py /app/
COPY meter_classifier.pth /app/
COPY main.py /app/
COPY data/ /app/data/


EXPOSE 5001

CMD ["python", "main.py"]
