import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO

# KNN utils
from knn.knn_utils import load_model_knn, predict_knn

# ===============================
# --- Argparse ------------------
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument('--yolo_model', required=True)
parser.add_argument('--source', required=True, help='File, folder, video, or camera (usb0, usb1)')
parser.add_argument('--knn_model', required=True, help='Path ke KNN pickle file')
parser.add_argument('--thresh', type=float, default=0.5)
parser.add_argument('--resolution', default=None, help='WxH, misal 640x480')
parser.add_argument('--record', action='store_true')
args = parser.parse_args()

# ===============================
# --- Load YOLO -----------------
# ===============================
yolo_model = YOLO(args.yolo_model, task='detect')
labels = yolo_model.names

# ===============================
# --- Load KNN ------------------
# ===============================
knn_model, knn_scaler = load_model_knn(args.knn_model)

# ===============================
# --- Setup source --------------
# ===============================
img_ext = ['.jpg','.jpeg','.png','.bmp']
vid_ext = ['.mp4','.avi','.mov','.mkv']

source_type = None
resize = False

if args.resolution:
    resize = True
    resW, resH = map(int, args.resolution.split('x'))

# Tentukan tipe source
if os.path.isdir(args.source):
    source_type = 'folder'
    imgs_list = [f for f in glob.glob(os.path.join(args.source, '*')) if os.path.splitext(f)[1].lower() in img_ext]
elif os.path.isfile(args.source):
    ext = os.path.splitext(args.source)[1].lower()
    if ext in img_ext:
        source_type = 'image'
        imgs_list = [args.source]
    elif ext in vid_ext:
        source_type = 'video'
        cap = cv2.VideoCapture(args.source)
    else:
        print(f'File {args.source} tidak didukung.')
        sys.exit(0)
elif 'usb' in args.source:
    source_type = 'usb'
    usb_idx = int(args.source.replace('usb',''))
    cap = cv2.VideoCapture(usb_idx)
elif 'picamera' in args.source:
    from picamera2 import Picamera2
    source_type = 'picamera'
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW,resH)}))
    cap.start()
else:
    print(f'Input {args.source} tidak valid.')
    sys.exit(0)

# ===============================
# --- Run YOLO + KNN ------------
# ===============================
img_count = 0
while True:
    # Ambil frame
    if source_type in ['image','folder']:
        if img_count >= len(imgs_list):
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type in ['video','usb']:
        ret, frame = cap.read()
        if not ret:
            break
    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            break

    # Resize jika diminta
    if resize:
        frame = cv2.resize(frame, (resW,resH))

    # YOLO inference
    results = yolo_model(frame, verbose=False)
    detections = results[0].boxes

    for det in detections:
        conf = det.conf.item()
        if conf < args.thresh:
            continue

        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        class_idx = int(det.cls.item())
        classname = labels[class_idx]

        # Crop untuk KNN
        crop = frame[ymin:ymax, xmin:xmax]
        knn_label, knn_prob = predict_knn(crop, knn_model, knn_scaler)

        # Draw YOLO bbox
        color = (0,255,0)
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
        cv2.putText(frame, f'{classname} ({conf*100:.1f}%)', (xmin,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Draw KNN label
        cv2.putText(frame, f'KNN: {knn_label}', (xmin, ymax+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.imshow('YOLO + KNN', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.imwrite('capture.png', frame)

# Clean up
if source_type in ['video','usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
cv2.destroyAllWindows()
