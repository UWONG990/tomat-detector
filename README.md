# YOLO + KNN Tomato Detection

## Dataset

### YOLO (Deteksi Objek)

Gunakan dataset YOLO untuk training model deteksi.

Link: https://drive.google.com/file/d/1RQSFlyxGrJLFjc3q6NY1vGeNbyPrLP7p/view?usp=drivesdk

### KNN (Klasifikasi)

Gunakan dataset KNN untuk ekstraksi fitur warna dan klasifikasi kematangan tomat.

Link: https://github.com/up2metric/tomatOD

## Setting Environment

Buat virtual environment:

```bash
python -m venv venv
```

Aktifkan virtual environment (Windows):

```bash
venv\Scripts\activate
```

Install dependency:

```bash
pip install opencv-python ultralytics numpy pandas scikit-learn joblib
```

## Menjalankan Program

Untuk menjalankan deteksi tomat real-time menggunakan kamera USB:

```bash
python yolo_knn.py --yolo_model my_model.pt --knn_model knn/train/models/knn_model.pkl --source usb0 --resolution 640x480
```

### Keterangan Parameter:

- `--yolo_model` : path ke model YOLO (.pt)
- `--knn_model` : path ke model KNN (.pkl)
- `--source usb0` : kamera USB default (ubah ke usb1 jika tidak terbuka)
- `--resolution` : resolusi kamera (lebar x tinggi)
