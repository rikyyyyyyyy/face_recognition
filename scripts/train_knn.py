import os
import pickle
from sklearn import neighbors
import face_recognition
from azure.storage.blob import BlobServiceClient

# Azure Blob Storageの設定
blob_service_client = BlobServiceClient.from_connection_string("Your_Connection_String")
container_name = "models"
container_client = blob_service_client.get_container_client(container_name)

def load_training_data(image_dir):
    X = []
    y = []
    
    # Azure Blob Storageから画像を取得
    blobs = container_client.list_blobs(name_starts_with=image_dir)
    for blob in blobs:
        blob_client = container_client.get_blob_client(blob)
        image_data = blob_client.download_blob().readall()
        image = face_recognition.load_image_file(image_data)
        
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            X.append(face_encodings[0])
            y.append(blob.name.split(".")[0])
    
    return X, y

def train_knn(X, y):
    if len(X) == 0:
        raise ValueError("トレーニングデータが空です。")
    
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(X, y)
    
    # モデルをローカルに保存し、Azure Blob Storageにアップロード
    model_path = "models/knn_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(knn_clf, f)
    
    blob_client = container_client.get_blob_client("knn_model.pkl")
    with open(model_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

if __name__ == "__main__":
    X, y = load_training_data("augmented_images")
    train_knn(X, y)
