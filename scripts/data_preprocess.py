import cv2
import face_recognition
import numpy as np
from albumentations import (
    HorizontalFlip, RandomRotate90, ShiftScaleRotate, 
    RandomBrightnessContrast, Compose
)
from azure.storage.blob import BlobServiceClient
import os

# Azure Blob Storageの設定
blob_service_client = BlobServiceClient.from_connection_string("Your_Connection_String")
container_name = "training-images"
container_client = blob_service_client.get_container_client(container_name)

def align_face(image):
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        raise ValueError("顔が検出されませんでした。")
    
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]
    face_image = cv2.resize(face_image, (150, 150))
    
    return face_image

def augment_image(image):
    aug = Compose([
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        RandomBrightnessContrast(p=0.2),
    ])
    augmented = aug(image=image)
    return augmented['image']

def process_images():
    blob_list = container_client.list_blobs()
    
    for blob in blob_list:
        blob_client = container_client.get_blob_client(blob.name)
        downloaded_blob = blob_client.download_blob().readall()
        
        image = cv2.imdecode(np.frombuffer(downloaded_blob, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        
        aligned_face = align_face(image)
        augmented_faces = [augment_image(aligned_face) for _ in range(5)]
        
        for i, aug_face in enumerate(augmented_faces):
            aug_image_name = f"augmented_{i}_{blob.name}"
            aug_image_path = os.path.join("augmented_images", aug_image_name)
            cv2.imwrite(aug_image_path, aug_face)
            
            # Azure Blob Storageにアップロード
            with open(aug_image_path, "rb") as data:
                container_client.upload_blob(aug_image_name, data, overwrite=True)
            os.remove(aug_image_path)

if __name__ == "__main__":
    process_images()
