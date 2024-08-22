import face_recognition
import pickle
import sys
import os
from azure.storage.blob import BlobServiceClient

# Azure Blob Storageの設定
blob_service_client = BlobServiceClient.from_connection_string("Your_Connection_String")
container_name = "models"
container_client = blob_service_client.get_container_client(container_name)

def load_model():
    model_path = "models/knn_model.pkl"
    if not os.path.exists(model_path):
        blob_client = container_client.get_blob_client("knn_model.pkl")
        with open(model_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
    with open(model_path, "rb") as f:
        return pickle.load(f)

knn_clf = load_model()

def predict_from_image(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    
    if len(face_encodings) == 0:
        print("顔が検出されませんでした。")
        return
    
    for face_encoding in face_encodings:
        name = knn_clf.predict([face_encoding])
        print(f"予測された名前: {name[0]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_from_image.py <image_file_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predict_from_image(image_path)
