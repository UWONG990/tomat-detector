import cv2
import numpy as np
import pandas as pd
import joblib
import os

def load_model_knn(model_path):
    """
    Load KNN model dan scaler dari file pickle
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} tidak ditemukan")
    
    data = joblib.load(model_path)
    model = data['model']
    scaler = data['scaler']
    return model, scaler

def extract_features(crop):
    """
    Ekstraksi fitur dari crop (RGB, HSV, rasio R/G & R/B)
    Masking piksel merah jika ada
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    
    if np.count_nonzero(mask) == 0:
        feature_crop = crop
    else:
        feature_crop = cv2.bitwise_and(crop, crop, mask=mask)
    
    B, G, R = feature_crop.mean(axis=(0,1))
    H, S, V = cv2.cvtColor(feature_crop, cv2.COLOR_BGR2HSV).mean(axis=(0,1))
    rg_ratio = R / (G + 1e-5)
    rb_ratio = R / (B + 1e-5)
    
    df = pd.DataFrame([[R, G, B, H, S, V, rg_ratio, rb_ratio]],
                      columns=["R","G","B","H","S","V","R/G","R/B"])
    return df

def predict_knn(crop, model, scaler):
    """
    Prediksi kelas tomat dari crop menggunakan KNN
    """
    X = extract_features(crop)
    X_scaled = scaler.transform(X)
    label = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    return label, prob
