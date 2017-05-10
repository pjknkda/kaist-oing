import logging
import os
import time

import numpy as np

import cv2

logging.basicConfig(
    format='[%(levelname)1.1s %(asctime)s P%(process)d] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

face_cascade = cv2.CascadeClassifier(
    os.path.join('cv2_data', 'haarcascades', 'haarcascade_frontalface_default.xml')
)
eye_cascade = cv2.CascadeClassifier(
    os.path.join('cv2_data', 'haarcascades', 'haarcascade_eye_tree_eyeglasses.xml')
)


def points_affine_transform(points, mat):
    return mat.dot(
        np.hstack([
            points,
            np.ones(shape=(points.shape[0], 1))
        ]).T
    ).T


def rotated_face_detection(degree, frame, gray):
    frame_h, frame_w, _ = frame.shape

    rotate_mat = cv2.getRotationMatrix2D((frame_w / 2, frame_h / 2), degree, 1)
    rotate_mat_inv = cv2.invertAffineTransform(rotate_mat)

    gray_rotated = cv2.warpAffine(gray, rotate_mat, (frame_w, frame_h))

    faces = face_cascade.detectMultiScale(gray_rotated, 1.3, 5)
    for x, y, w, h in faces:
        face_rect_points = points_affine_transform(
            np.array([[x, y],
                      [x + w, y],
                      [x + w, y + h],
                      [x, y + h]]),
            rotate_mat_inv
        )
        cv2.polylines(frame, [np.int32(face_rect_points)], True, (255, 0, 0), 2)

        eyes = eye_cascade.detectMultiScale(gray_rotated[y:y + h, x:x + w])
        for ex, ey, ew, eh in eyes:
            eyes_rect_points = points_affine_transform(
                np.array([[x + ex, y + ey],
                          [x + ex + ew, y + ey],
                          [x + ex + ew, y + ey + eh],
                          [x + ex, y + ey + eh]]),
                rotate_mat_inv
            )
            cv2.polylines(frame, [np.int32(eyes_rect_points)], True, (0, 255, 0), 2)


def tilted_face_detection(pts1, pts2, frame):
    frame_h, frame_w, _ = frame.shape

    M = cv2.getPerspectiveTransform(pts1, pts2)
    M_inv = cv2.getPerspectiveTransform(pts2, pts1)

    frame_tilted = cv2.warpPerspective(frame, M, (frame_w, frame_h))

    gray_tilted = cv2.cvtColor(frame_tilted, cv2.COLOR_BGR2GRAY)
    for degree in [-30, 0, 30]:
        rotated_face_detection(degree, frame_tilted, gray_tilted)

    return cv2.warpPerspective(frame_tilted, M_inv, (frame_w, frame_h))


def main():
    cap = cv2.VideoCapture(0)

    p1 = 1 / 3
    p2 = 1 - p1

    while True:
        is_success, frame = cap.read()
        if not is_success:
            continue

        frame_h, frame_w, _ = frame.shape

        pts1 = np.float32([[0, 0], [frame_w, 0], [frame_w, frame_h], [0, frame_h]])
        pts2 = np.float32([[0, 0], [frame_w, int(frame_h * p1)], [frame_w, int(frame_h * p2)], [0, frame_h]])
        pts3 = np.float32([[0, int(frame_h * p1)], [frame_w, 0], [frame_w, frame_h], [0, int(frame_h * p2)]])

        # r_right = tilted_face_detection(pts1, pts2, frame)
        r_middle = tilted_face_detection(pts1, pts1, frame)
        # r_left = tilted_face_detection(pts1, pts3, frame)

        # cv2.imshow('left', r_left)
        cv2.imshow('middle', r_middle)
        # cv2.imshow('right', r_right)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(1 / 30)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
