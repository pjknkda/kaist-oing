import functools
import io
import logging
import os
import threading
import time

import numpy as np

import cv2
from google.cloud import vision
from google.oauth2 import service_account

logging.basicConfig(
    format='[%(levelname)1.1s %(asctime)s P%(process)d] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


logger.info('Hello')

vision_client = vision.Client(
    project='Oing',
    credentials=service_account.Credentials.from_service_account_file(
        'Oing-9f81400683ae.json'
    )
)

FACE_DETECTION_INTERVAL = 5.0
FACE_DETECTION_MAX_FACES = 32


def _detect_faces(detection_info):
    img = vision_client.image(filename='_temp.jpg')
    detection_info['result'] = img.detect_faces(limit=FACE_DETECTION_MAX_FACES)
    detection_info['last_ts'] = time.time()
    detection_info['is_waiting'] = False


def _pos_to_tuple(pos):
    return (int(pos.x_coordinate), int(pos.y_coordinate))


def main():
    cap = cv2.VideoCapture(0)

    detection_info = {
        'last_ts': time.time(),
        'is_waiting': False,
        'result': None
    }

    while cap.isOpened():
        is_success, frame = cap.read()
        if not is_success:
            continue

        frame_h, frame_w, _ = frame.shape

        if (not detection_info['is_waiting']
                and FACE_DETECTION_INTERVAL < time.time() - detection_info['last_ts']):
            detection_info['is_waiting'] = True
            cv2.imwrite('_temp.jpg', frame)
            threading.Thread(target=functools.partial(_detect_faces, detection_info)).start()

        if detection_info['result'] is not None:
            for face in detection_info['result']:
                # Label for angle information
                cv2.putText(
                    frame,
                    'pan %.2f, roll %.2f, tilt %.2f' % (face.angles.pan,
                                                        face.angles.roll,
                                                        face.angles.tilt),
                    (  # origin
                        face.bounds.vertices[0].x_coordinate,
                        face.bounds.vertices[0].y_coordinate - 10
                    ),
                    cv2.FONT_HERSHEY_DUPLEX,  # font face
                    0.6,  # font scale
                    (255, 0, 0)  # color
                )

                # Left eye (Person's right eye)
                cv2.circle(
                    frame,
                    _pos_to_tuple(face.landmarks.left_eye.position),
                    2,  # radius
                    (0, 255, 0),  # color
                    -2  # thickness
                )

                # Right eye (Person's left eye)
                cv2.circle(
                    frame,
                    _pos_to_tuple(face.landmarks.right_eye.position),
                    2,  # radius
                    (0, 255, 0),  # color
                    -2  # thickness
                )

                # Face boundary
                cv2.polylines(
                    frame,
                    [
                        np.array([_pos_to_tuple(vtx) for vtx in face.bounds.vertices])
                    ],
                    True,  # is_closed
                    (255, 0, 0),  # color
                    2  # thickness
                )

        cv2.imshow('Oing', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
