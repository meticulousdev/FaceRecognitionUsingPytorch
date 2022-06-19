import cv2
import numpy as np
from typing import List


def face_recognition_video(file_name: str,
                           face_model, face_config,
                           onnx_model_path: str, classes: List[str]) -> None:
    src: cv2.VideoCapture = cv2.VideoCapture(file_name)

    # face_detection:
    face_net: cv2.dnn_Net = cv2.dnn.readNet(face_model, face_config)
    onnx_net = cv2.dnn.readNetFromONNX(onnx_model_path)

    while True:
        ret, frame = src.read()
        if ret is None:
            continue

        blob: np.ndarray = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
        face_net.setInput(blob)
        out: np.ndarray = face_net.forward()

        detect: np.ndarray = out[0, 0, :, :]
        (h, w) = frame.shape[:2]

        for i in range(detect.shape[0]):
            confidence: np.ndarray = detect[i, 2]
            if confidence < 0.5:
                break

            x1: int = int(detect[i, 3] * w)
            y1: int = int(detect[i, 4] * h)
            x2: int = int(detect[i, 5] * w)
            y2: int = int(detect[i, 6] * h)

            # face detection and extraction
            detected_face: np.ndarray = frame[y1:y2, x1:x2].copy()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

            blob = cv2.dnn.blobFromImage(detected_face, 1.0 / 255, (100, 100), (104, 177, 123))
            onnx_net.setInput(blob)
            preds = onnx_net.forward()

            biggest_pred_index = np.array(preds)[0].argmax()

            label = f'{classes[biggest_pred_index]}: {confidence:4.2f}'
            cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    onnx_model_path: str = './onnx_model/pytorch_face_recognition.onnx'

    face_model: str = 'opencv_face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    face_config: str = 'opencv_face_detector/deploy.prototxt'

    classes: List[str] = ['IU', 'madclown', 'sunmi']

    # file_name = './video/person_test_sunmi_02.mp4'
    file_name = './video/person_test_IU_02.mp4'
    # file_name = './video/person_test_madclown_01.mp4'
    face_recognition_video(file_name, face_model, face_config, onnx_model_path, classes)
