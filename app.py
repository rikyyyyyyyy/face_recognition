from flask import Flask, request, jsonify, render_template
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError  # ここでインポート
import os
import pickle
import face_recognition

app = Flask(__name__)

# Azure Blob Storageの接続設定
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = "models"
container_client = blob_service_client.get_container_client(container_name)

# コンテナーが存在しない場合に作成する
try:
    container_client.create_container()
    print(f"コンテナー '{container_name}' を作成しました。")
except ResourceExistsError:  # ここで正しく例外処理を行う
    print(f"コンテナー '{container_name}' はすでに存在しています。")

# モデルのロード
def load_model():
    model_path = "models/knn_model.pkl"
    if not os.path.exists(model_path):
        blob_client = container_client.get_blob_client("knn_model.pkl")
        with open(model_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
    with open(model_path, "rb") as f:
        return pickle.load(f)

knn_clf = load_model()

# 他のルート定義やFlaskのコードが続く...

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['file']
    image_path = os.path.join("uploads", image.filename)
    image.save(image_path)

    img = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(img)

    if len(face_encodings) == 0:
        return jsonify({"error": "No faces detected."}), 400

    predictions = []
    for face_encoding in face_encodings:
        name = knn_clf.predict([face_encoding])[0]
        predictions.append(name)

    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
