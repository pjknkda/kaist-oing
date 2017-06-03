import collections
import csv
import glob
import os.path

import numpy as np

import cv2

DATA_BASE_DIR = 'D:\Downloads\Oing_data'
IMG_DIR_NAME_FORMAT = '170524_%s_101'
DETECTION_NAME_FORMAT = 'detection_%s.csv'

IMAGE_SIZE = (1024, 768, 768)  # (x, y, z)
Z_INFO = {
    'unit_face_length': 100,
    'unit_z_pos': 500,
    'padding': 250
}
GAUSSIAN_SIZE = (311, 311)
MAP_CUMMULATE_LENGTH = 50
MAP_CUMMULATE_LIMIT = 15
MAP_FACE_SIZE = (30, 30)  # x, z


def bound_to_rect(face, ratio_x=1.0, ratio_y=1.0):
    return np.array([
        (face['bound_left'] * ratio_x, face['bound_top'] * ratio_y),
        (face['bound_right'] * ratio_x, face['bound_top'] * ratio_y),
        (face['bound_right'] * ratio_x, face['bound_bot'] * ratio_y),
        (face['bound_left'] * ratio_x, face['bound_bot'] * ratio_y)
    ], dtype=int)


face_map_list = []


def convert_to_3d(img, faces):
    img_h, img_w, _ = img.shape

    img_orig = img
    img = cv2.resize(img, IMAGE_SIZE[:2], interpolation=cv2.INTER_AREA)

    ratio_x = IMAGE_SIZE[0] / img_w
    ratio_y = IMAGE_SIZE[1] / img_h

    face_cords = []

    # Draw rectangles on the faces
    for face in faces:
        face_length = max(face['bound_right'] - face['bound_left'],  face['bound_top'] - face['bound_bot'])

        face_x = int((face['bound_right'] + face['bound_left']) / 2 * ratio_x)
        face_y = int((face['bound_top'] + face['bound_bot']) / 2 * ratio_y)
        face_z = max(
            0,
            min(
                IMAGE_SIZE[2],
                int(Z_INFO['unit_z_pos'] * (Z_INFO['unit_face_length'] / face_length)) - Z_INFO['padding']
            )
        )

        face_cords.append((face_x, face_y, face_z))

        cv2.putText(
            img,
            '%d, %d, %d' % (face_x, face_y, face_z),
            (  # origin
                int(face['bound_left'] * ratio_x),
                int(face['bound_top'] * ratio_y) - 10
            ),
            cv2.FONT_HERSHEY_DUPLEX,  # font face
            0.3,  # font scale
            (255, 0, 0)  # color
        )

        cv2.polylines(
            img,
            [bound_to_rect(face, ratio_x, ratio_y)],
            True,  # is_closed
            (255, 0, 0),  # color
            2  # thickness
        )

    face_cords = np.array(face_cords)
    face_map = np.zeros((IMAGE_SIZE[2] + 1, IMAGE_SIZE[0] + 1))

    for face_cord in face_cords:
        face_map[IMAGE_SIZE[2] - face_cord[2], face_cord[0]] = 255

    face_map = cv2.GaussianBlur(face_map, GAUSSIAN_SIZE, 0)  # TODO : Gaussian is not proper
    face_map_list.append(face_map)

    face_map_averaged = np.zeros((IMAGE_SIZE[2] + 1, IMAGE_SIZE[0] + 1))
    for i_face_map in face_map_list[-MAP_CUMMULATE_LENGTH:]:
        face_map_averaged += i_face_map
    face_map_averaged = np.clip(face_map_averaged, 0, MAP_CUMMULATE_LIMIT * 3)

    face_map_averaged *= (1 / np.max(face_map_averaged)) * 255
    face_map_averaged = np.array(face_map_averaged, dtype=np.uint8)

    face_map_colored = cv2.applyColorMap(face_map_averaged, cv2.COLORMAP_HOT)

    for idx, face in enumerate(faces):
        face = img_orig[face['bound_top']:face['bound_bot'],
                        face['bound_left']:face['bound_right']]

        face = cv2.resize(face, MAP_FACE_SIZE)
        x = face_cords[idx][0]
        z = IMAGE_SIZE[2] - face_cords[idx][2]
        face_map_colored[z - int(MAP_FACE_SIZE[1] / 2):z + int(MAP_FACE_SIZE[1] / 2),
                         x - int(MAP_FACE_SIZE[0] / 2):x + int(MAP_FACE_SIZE[0] / 2)] = face

    cv2.imshow('Oing', img)  # x-y map
    cv2.imshow('Heatmap', face_map_colored)  # x-z map


def load_dataset(name):
    img_path_list = sorted(
        glob.iglob(
            os.path.join(DATA_BASE_DIR, IMG_DIR_NAME_FORMAT % name,  '*.JPG')
        )
    )

    face_dict = collections.defaultdict(list)
    with open(os.path.join(DATA_BASE_DIR, DETECTION_NAME_FORMAT % name), 'r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            for unused_field in ('dir_name', 'anger', 'joy', 'sorrow', 'surprise'):
                del row[unused_field]

            for int_field in ('bound_left', 'bound_top', 'bound_right', 'bound_bot'):
                row[int_field] = int(row[int_field])

            for float_field in ('pan', 'roll', 'tilt'):
                row[float_field] = float(row[float_field])

            face_dict[row.pop('img_file').lower()].append(row)

    for img_path in img_path_list:
        img = cv2.imread(img_path)
        faces = face_dict[os.path.basename(img_path).lower()]

        convert_to_3d(img, faces)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


load_dataset('0900')
# load_dataset('1030')
# load_dataset('0900')
