FROM python:3.9-slim

# CMakeやビルドに必要なツールをインストール
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    && apt-get clean

WORKDIR /app

COPY . /app

# 依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
