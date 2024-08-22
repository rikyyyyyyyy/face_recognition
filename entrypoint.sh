#!/bin/bash

if [ "$1" = "preprocess" ]; then
    python scripts/data_preprocess.py
elif [ "$1" = "train" ]; then
    python scripts/train_knn.py
elif [ "$1" = "predict" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 predict <image_file_path>"
        exit 1
    fi
    python scripts/predict_from_image.py "$2"
else
    echo "Usage: $0 {preprocess|train|predict}"
    exit 1
fi
