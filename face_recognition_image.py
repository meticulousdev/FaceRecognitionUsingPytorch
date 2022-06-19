import cv2
import numpy as np

onnx_model_path = './onnx_model/pytorch_face_recognition.onnx'

face_model: str = 'opencv_face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
face_config: str = 'opencv_face_detector/deploy.prototxt'

classes = ['IU', 'madclown', 'sunmi']

img_path = './image/'
img_files = ['madclown_01.jpg', 'madclown_02.jpg', 'IU_01.jpg', 'IU_02.jpg', 'sunmi_01.jpg', 'sunmi_02.jpg']
# face_detection:

face_net: cv2.dnn_Net = cv2.dnn.readNet(face_model, face_config)
onnx_net = cv2.dnn.readNetFromONNX(onnx_model_path)

for f in img_files:
    img = cv2.imread(img_path + f)
    if img is None:
        continue

    blob: np.ndarray = cv2.dnn.blobFromImage(img, 1, (300, 300), (104, 177, 123))
    face_net.setInput(blob)
    out: np.ndarray = face_net.forward()

    detect: np.ndarray = out[0, 0, :, :]
    (h, w) = img.shape[:2]

    confidence: np.ndarray = detect[0, 2]

    x1: int = int(detect[0, 3] * w)
    y1: int = int(detect[0, 4] * h)
    x2: int = int(detect[0, 5] * w)
    y2: int = int(detect[0, 6] * h)

    # face detection and extraction
    detected_face: np.ndarray = img[y1:y2, x1:x2].copy()

    while True:
        cv2.imshow('detected_face', detected_face)

        if cv2.waitKey(1) == 27:
            break

    blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (100, 100), (104, 177, 123))
    onnx_net.setInput(blob)
    preds = onnx_net.forward()
    print(preds[0])

    biggest_pred_index = np.array(preds)[0].argmax()
    print("Predicted class:", classes[biggest_pred_index])

cv2.destroyAllWindows()
