import collections
import csv
import glob
import os.path

import numpy as np

import cv2

DATA_BASE_DIR = './data'
IMG_DIR_NAME_FORMAT = '170524_%s_101'
DETECTION_NAME_FORMAT = 'detection_%s.csv'

# IMAGE_SIZE = (320, 240, 768)  # (x, y, z)
IMAGE_SIZE = (1024, 768, 768)  # (x, y, z)
# IMAGE_SIZE = (640, 480, 1)  # (x, y, z)


def bound_to_rect(face, ratio_x=1.0, ratio_y=1.0):
    return np.array([
        (face['bound_left'] * ratio_x, face['bound_top'] * ratio_y),
        (face['bound_right'] * ratio_x, face['bound_top'] * ratio_y),
        (face['bound_right'] * ratio_x, face['bound_bot'] * ratio_y),
        (face['bound_left'] * ratio_x, face['bound_bot'] * ratio_y)
    ], dtype=int)


face_map_list = []
face_map_long_list = []
img_fg_list = []
fgbg = cv2.createBackgroundSubtractorMOG2()


def img_overlay_high(img, faces, idx):
    global face_map_list
    global img_fg_list

    FACE_PADDING_RATIO = 0.15

    MAP_FACE_SIZE = (60, 60)
    MAP_GAUSSIAN_SIZE = (55, 55)
    MAP_CUMMULATE_LENGTH = 50
    MAP_CUMMULATE_LIMIT = 5

    FG_GAUSSIAN_SIZE = (55, 55)
    FG_CUMMULATE_LENGTH = 50

    img_h, img_w, _ = img.shape

    ratio_x = IMAGE_SIZE[0] / img_w
    ratio_y = IMAGE_SIZE[1] / img_h

    img_orig = img
    img = cv2.resize(img, IMAGE_SIZE[:2], interpolation=cv2.INTER_AREA)

    img_fg = fgbg.apply(img)
    img_fg = cv2.GaussianBlur(img_fg, FG_GAUSSIAN_SIZE, 0)
    img_fg[np.where(img_fg > 0)] = 1
    img_fg_list.append(img_fg)
    img_fg_list = img_fg_list[-FG_CUMMULATE_LENGTH:]

    face_map = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    for face in faces:
        face_h = face['bound_bot'] - face['bound_top']
        face_w = face['bound_right'] - face['bound_left']

        top = max(0, int(face['bound_top'] * ratio_y - face_h * FACE_PADDING_RATIO))
        bot = min(IMAGE_SIZE[1], int(face['bound_bot'] * ratio_y + face_h * FACE_PADDING_RATIO))
        left = max(0, int(face['bound_left'] * ratio_x - face_w * FACE_PADDING_RATIO))
        right = min(IMAGE_SIZE[0], int(face['bound_right'] * ratio_x + face_w * FACE_PADDING_RATIO))

        face_map[top:bot, left:right] = 255

        cv2.polylines(
            img,
            [bound_to_rect(face, ratio_x, ratio_y)],
            True,  # is_closed
            (255, 0, 0),  # color
            1  # thickness
        )

        # face_orig = img_orig[face['bound_top']:face['bound_bot'],
        #                      face['bound_left']:face['bound_right']]
        # face_orig = cv2.resize(face_orig, MAP_FACE_SIZE)
        # img[int((top + bot) / 2 - MAP_FACE_SIZE[1] / 2):
        #     int((top + bot) / 2 + MAP_FACE_SIZE[1] / 2),
        #     int((left + right) / 2 - MAP_FACE_SIZE[0] / 2):
        #     int((left + right) / 2 + MAP_FACE_SIZE[0] / 2)] = face_orig

    face_map = cv2.GaussianBlur(face_map, MAP_GAUSSIAN_SIZE, -0.5)  # TODO : Gaussian may not be proper
    face_map = np.array(face_map, dtype=np.uint8)
    face_map_list.append(face_map)
    face_map_list = face_map_list[-MAP_CUMMULATE_LENGTH:]

    face_map_overlay_averaged = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    for iter_face_map in face_map_list:
        face_map_overlay_averaged += iter_face_map
    face_map_overlay_averaged /= MAP_CUMMULATE_LIMIT

    face_map_overaly = np.dstack([face_map_overlay_averaged] * 3)
    face_map_overaly = np.array(face_map_overaly, dtype=np.float) / 255
    face_map_overaly = np.clip(face_map_overaly, 0.1, 1)

    img_fg_overlay = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    for iter_img_fg in img_fg_list:
        img_fg_overlay += iter_img_fg
    img_fg_overlay[np.where(img_fg_overlay < np.mean(img_fg_overlay))] = 0
    img_fg_overlay[np.where(img_fg_overlay > 0)] = 1
    img_fg_overlay = np.dstack([img_fg_overlay] * 3)

    overlay = face_map_overaly * img_fg_overlay + 0.05
    # overlay = (1 - face_map_overaly) * img_fg_overlay + 0.2
    overlay = np.clip(overlay, 0, 1)

    overlay_heatmap = np.array(overlay * 255, dtype=np.uint8)
    overlay_heatmap = cv2.applyColorMap(overlay_heatmap, cv2.COLORMAP_JET)

    merged_img = np.array(img * overlay, dtype=np.uint8)

    cv2.putText(
        merged_img,
        'Idx %d' % idx,
        (50, 50),
        cv2.FONT_HERSHEY_DUPLEX,  # font face
        0.5,  # font scale
        (0, 255, 255)  # color
    )
    cv2.imshow('Oing', merged_img)
    # cv2.imshow('Oing Heatmap', overlay_heatmap)


def img_overlay_low(img, faces, idx):
    global face_map_list
    global face_map_long_list

    FACE_PADDING_RATIO = 0.25

    MAP_FACE_SIZE = (60, 60)
    MAP_GAUSSIAN_SIZE = (55, 55)
    MAP_CUMMULATE_LENGTH = 60

    LONG_CUMMULATE_LENGTH = 300

    img_h, img_w, _ = img.shape

    ratio_x = IMAGE_SIZE[0] / img_w
    ratio_y = IMAGE_SIZE[1] / img_h

    img_orig = img
    img = cv2.resize(img, IMAGE_SIZE[:2], interpolation=cv2.INTER_AREA)

    face_map = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.bool)
    for face in faces:
        face_h = face['bound_bot'] - face['bound_top']
        face_w = face['bound_right'] - face['bound_left']

        top = max(0, int(face['bound_top'] * ratio_y - face_h * FACE_PADDING_RATIO))
        bot = min(IMAGE_SIZE[1], int(face['bound_bot'] * ratio_y + face_h * FACE_PADDING_RATIO))
        left = max(0, int(face['bound_left'] * ratio_x - face_w * FACE_PADDING_RATIO))
        right = min(IMAGE_SIZE[0], int(face['bound_right'] * ratio_x + face_w * FACE_PADDING_RATIO))

        face_map[top:bot, left:right] = 1

        cv2.polylines(
            img,
            [bound_to_rect(face, ratio_x, ratio_y)],
            True,  # is_closed
            (255, 0, 0),  # color
            1  # thickness
        )

    face_map_list.append(face_map)
    face_map_list = face_map_list[-MAP_CUMMULATE_LENGTH:]
    cummulative_face_map = np.any(face_map_list, axis=0)

    face_map_long_list.append(face_map)
    face_map_long_list = face_map_long_list[-LONG_CUMMULATE_LENGTH:]
    cummulative_face_long_map = np.any(face_map_long_list, axis=0)

    overlay_mask = np.logical_not(
        np.logical_and(np.logical_not(cummulative_face_map),
                       cummulative_face_long_map)
    )
    overlay = np.ones((IMAGE_SIZE[1], IMAGE_SIZE[0])) * 255
    overlay[overlay_mask] = 0

    overlay = cv2.GaussianBlur(overlay, MAP_GAUSSIAN_SIZE, -0.5)  # TODO : Gaussian may not be proper
    overlay = np.array(overlay, dtype=np.float) / 255
    overlay = np.clip(overlay, 0.1, 1)
    overlay = np.dstack([overlay] * 3)

    merged_img = np.array(img * overlay, dtype=np.uint8)

    cv2.putText(
        merged_img,
        'Idx %d' % idx,
        (50, 50),
        cv2.FONT_HERSHEY_DUPLEX,  # font face
        0.5,  # font scale
        (0, 255, 255)  # color
    )
    cv2.imshow('Oing', merged_img)


def convert_to_3d(img, faces, idx):
    global face_map_list

    Z_INFO = {
        'unit_face_length': 100,
        'unit_z_pos': 500,
        'padding': 250
    }
    GAUSSIAN_SIZE = (311, 311)
    MAP_CUMMULATE_LENGTH = 50
    MAP_CUMMULATE_LIMIT = 5
    MAP_FACE_SIZE = (30, 30)  # x, z

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

    face_map = cv2.GaussianBlur(face_map, GAUSSIAN_SIZE, 0)  # TODO : Gaussian may not be proper
    face_map_list.append(face_map)
    face_map_list = face_map_list[-MAP_CUMMULATE_LENGTH:]

    face_map_averaged = np.zeros((IMAGE_SIZE[2] + 1, IMAGE_SIZE[0] + 1))
    for iter_face_map in face_map_list:
        face_map_averaged += iter_face_map
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
        try:
            face_map_colored[z - int(MAP_FACE_SIZE[1] / 2):z + int(MAP_FACE_SIZE[1] / 2),
                             x - int(MAP_FACE_SIZE[0] / 2):x + int(MAP_FACE_SIZE[0] / 2)] = face
        except ValueError:
            pass  # sometimes, "x" can exceed the image boundary

    cv2.putText(
        img,
        'Idx %d' % idx,
        (50, 50),
        cv2.FONT_HERSHEY_DUPLEX,  # font face
        0.5,  # font scale
        (0, 255, 255)  # color
    )
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
            if row.get('label') in ('1', '3'):
                continue

            for unused_field in ('dir_name', 'anger', 'joy', 'sorrow', 'surprise'):
                del row[unused_field]

            for int_field in ('bound_left', 'bound_top', 'bound_right', 'bound_bot'):
                row[int_field] = int(row[int_field])

            for float_field in ('pan', 'roll', 'tilt'):
                row[float_field] = float(row[float_field])

            face_dict[row.pop('img_file').lower()].append(row)

    for idx, img_path in enumerate(img_path_list):
        img = cv2.imread(img_path)
        faces = face_dict[os.path.basename(img_path).lower()]

        img_overlay_low(img, faces, idx)
        # img_overlay_high(img, faces, idx)
        # convert_to_3d(img, faces, idx)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


load_dataset('0900')
# load_dataset('1030')
# load_dataset('1430')
