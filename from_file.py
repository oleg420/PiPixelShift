import argparse
import os
import glob

import cv2

from utils import read_raw_8bit, pixel_shift
from consts import CameraVersion

def arg2camver(arg):
    if int(arg) == CameraVersion.V1:
        return CameraVersion.V1
    elif int(arg) == CameraVersion.V2:
        return CameraVersion.V2
    else:
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PiPixelShift. Read 4 images and create 1 pixel-shift image')
    parser.add_argument('-i', '--images', required=True, type=str)
    parser.add_argument('-v', '--camera_version', type=arg2camver, default=CameraVersion.V1)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.images):
        raise FileNotFoundError

    assert args.camera_version, 'Invalid camera version'

    paths = glob.glob(os.path.normpath(args.images) + '/*.jpeg')
    images = []
    for path in paths:
        images.append((read_raw_8bit(path, camera_version=args.camera_version)))

    pixel_shift_image = pixel_shift(images)

    cv2.imwrite('pixel_shift_image.png', pixel_shift_image)
    cv2.imshow('Pixel shift image', pixel_shift_image)
    cv2.waitKey()
    cv2.destroyAllWindows()