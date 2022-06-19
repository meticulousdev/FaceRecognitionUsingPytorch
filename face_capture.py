import cv2
import numpy as np
from enum import Enum

from typing import List
import os


class source_type(Enum):
    video = 1
    camera = 2


def face_capture(model: str, config: str, src_type: source_type,
                 data_type: str, img_name: str, case_num: str,
                 img_cnt: int = 100, file_name: str = 'none') -> bool:
    # source
    if src_type == source_type.video:
        src: cv2.VideoCapture = cv2.VideoCapture(file_name)
        if not src.isOpened():
            print('Video load failed!')
            return False

    elif src_type == source_type.camera:
        src: cv2.VideoCapture = cv2.VideoCapture(0)
        if not src.isOpened():
            print('Camera open failed!')
            return False

    else:
        print('Video or Camera?')
        return False

    # destination
    path: str = './capture/' + data_type + '/' + img_name + '/'
    img_file: str = path + img_name + '_' + case_num + '_'
    file_extension: str = '.jpg'
    if not os.path.isdir(path):
        os.mkdir(path)

    # deep neural network
    net: cv2.dnn_Net = cv2.dnn.readNet(model, config)
    if net.empty():
        print('Net open failed!')
        return False

    cnt: int = 0
    while True:
        ret, frame = src.read()
        if not ret:
            break

        blob: np.ndarray = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
        net.setInput(blob)
        out: np.ndarray = net.forward()

        detect: np.ndarray = out[0, 0, :, :]
        (h, w) = frame.shape[:2]

        for i in range(detect.shape[0]):
            confidence: np.ndarray = detect[i, 2]
            if confidence < 0.5:
                break

            cnt += 1
            x1: int = int(detect[i, 3] * w)
            y1: int = int(detect[i, 4] * h)
            x2: int = int(detect[i, 5] * w)
            y2: int = int(detect[i, 6] * h)

            # face detection and extraction
            detected_face: np.ndarray = frame[y1:y2, x1:x2].copy()
            # temp_face: np.ndarray = frame[y1:y2, x1:x2].copy()
            # detected_face: np.ndarray = cv2.resize(temp_face, dsize=(200, 150),
            #                                        interpolation=cv2.INTER_AREA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

            label = f'Face: {confidence:4.2f}'
            cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow('detected_face', detected_face)
            # des_path: str = img_file + str(cnt) + file_extension
            # cv2.imwrite(des_path, detected_face)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            break

        if cnt >= img_cnt:
            print(f'{cnt} {img_name} images for {data_type} are extracted')
            print('End of Extraction')
            break
        elif (cnt > 0) and (cnt % 1000 == 0):
            print(f'{cnt} {img_name} images for {data_type} are extracted')

    cv2.destroyAllWindows()
    return True
# end of def face_capture


if __name__ == "__main__":
    model: str = 'opencv_face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    config: str = 'opencv_face_detector/deploy.prototxt'

    img_cnt: int = 100
    name_list: List[str] = ['IU', 'sunmi', 'madclown']
    cases: List[str] = ['01', '02', '03']
    data_type_list: List[str] = ['train', 'test']

    for i in range(len(name_list)):
        for j in range(len(data_type_list)):
            for k in range(len(cases)):
                if j == 0:
                    idx = k
                else:
                    idx = 0

                file_ext: str = '.mp4'
                file_name: str = 'video/person_' + data_type_list[j]
                file_name = file_name + '_' + name_list[i] + '_' + cases[idx] + file_ext
                ret: bool = face_capture(model, config, source_type.video,
                                         data_type_list[j], name_list[i], cases[idx],
                                         img_cnt, file_name)

                if not ret:
                    print('Face capture failed!')
                    print(f'{name_list[i]}, {data_type_list[j]}')
