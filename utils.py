import cv2
import numpy as np

from consts import offset, reshape, crop, CameraVersion

def read_raw_16bit(path, camera_version=CameraVersion.V1):
    img = np.fromfile(path, dtype=np.uint8)
    img = img[-offset[camera_version]:]
    img = img[32768:]
    img = img.reshape(reshape[camera_version])[:crop[camera_version][0], :crop[camera_version][1]]

    img = img.astype(np.uint16) << 2
    for byte in range(4):
        img[:, byte::5] |= ((img[:, 4::5] >> ((4 - byte) * 2)) & 0b11)
    img = np.delete(img, np.s_[4::5], 1)

    return img


def read_raw_8bit(path, camera_version=CameraVersion.V1):
    img = np.fromfile(path, dtype=np.uint8)
    img = img[-offset[camera_version]:]
    img = img[32768:]
    img = img.reshape(reshape[camera_version])[:crop[camera_version][0], :crop[camera_version][1]]

    img = img.astype(np.uint16) << 2
    for byte in range(4):
        img[:, byte::5] |= ((img[:, 4::5] >> ((4 - byte) * 2)) & 0b11)
    img = np.delete(img, np.s_[4::5], 1)

    lower_bound = np.min(img)
    upper_bound = np.max(img)

    lut = np.concatenate([
        np.zeros(lower_bound, dtype=np.uint16),
        np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
        np.ones(2 ** 16 - upper_bound, dtype=np.uint16) * 255
    ])

    return lut[img].astype(np.uint8)

def pixel_shift(images):
    assert len(images) == 4, 'Not enough images'
    assert images[0].shape == images[1].shape == images[2].shape == images[3].shape, 'Shape are not equals'

    pixel_shift_img_raw = np.zeros((images[0].shape[0] - 1, images[0].shape[1] - 1, 4), dtype=np.uint8)

    pixel_shift_img_raw[:, :, 0] = images[0][:images[0].shape[0] - 1, :images[0].shape[1] - 1]
    pixel_shift_img_raw[:, :, 1] = images[1][:images[1].shape[0] - 1, 1:]
    pixel_shift_img_raw[:, :, 2] = images[2][1:, :images[2].shape[1] - 1]
    pixel_shift_img_raw[:, :, 3] = images[3][1:, 1:]

    # delete last row and col of pixel_shift_img
    pixel_shift_img_raw = pixel_shift_img_raw[:pixel_shift_img_raw.shape[0] - 1, :pixel_shift_img_raw.shape[1] - 1]

    # create pixel_shift_img
    pixel_shift_img = np.zeros((pixel_shift_img_raw.shape[0], pixel_shift_img_raw.shape[1], 3), dtype=np.uint8)

    pixel_shift_img[0::2, 0::2, 2] = pixel_shift_img_raw[0::2, 0::2, 2]
    pixel_shift_img[0::2, 0::2, 1] = np.minimum(pixel_shift_img_raw[0::2, 0::2, 0], pixel_shift_img_raw[0::2, 0::2, 3])
    pixel_shift_img[0::2, 0::2, 0] = pixel_shift_img_raw[0::2, 0::2, 1]

    pixel_shift_img[0::2, 1::2, 2] = pixel_shift_img_raw[0::2, 1::2, 3]
    pixel_shift_img[0::2, 1::2, 1] = np.minimum(pixel_shift_img_raw[0::2, 1::2, 1], pixel_shift_img_raw[0::2, 1::2, 2])
    pixel_shift_img[0::2, 1::2, 0] = pixel_shift_img_raw[0::2, 1::2, 0]

    pixel_shift_img[1::2, 0::2, 2] = pixel_shift_img_raw[1::2, 0::2, 0]
    pixel_shift_img[1::2, 0::2, 1] = np.minimum(pixel_shift_img_raw[1::2, 0::2, 1], pixel_shift_img_raw[1::2, 0::2, 2])
    pixel_shift_img[1::2, 0::2, 0] = pixel_shift_img_raw[1::2, 0::2, 3]

    pixel_shift_img[1::2, 1::2, 2] = pixel_shift_img_raw[1::2, 1::2, 1]
    pixel_shift_img[1::2, 1::2, 1] = np.minimum(pixel_shift_img_raw[1::2, 1::2, 0], pixel_shift_img_raw[1::2, 1::2, 3])
    pixel_shift_img[1::2, 1::2, 0] = pixel_shift_img_raw[1::2, 1::2, 2]

    return pixel_shift_img


def adjust_gamma(image, gamma=1.0):
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def wb_manual(img, temp, tint):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    if tint > 0:
        img[:, :, 1] += tint
    else:
        img[:, :, 1] += abs(tint)

    if temp > 0:
        img[:, :, 2] += temp
    else:
        img[:, :, 2] -= abs(temp)

    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    return img


def add_contrast(img, contrast=1, brightness=0):
    img[:, :, :] = np.clip(contrast * img[:, :, :] + brightness, 0, 255)

    return img


def add_saturation(img, value):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    (h, s, v) = cv2.split(img)

    s = np.clip(s * value, 0, 255)

    img = cv2.merge([h, s, v])
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img
