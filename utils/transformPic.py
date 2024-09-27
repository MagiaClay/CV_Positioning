import cv2
import os
import numpy as np

# 以原点反转
def flip_upsidedown(file_pathname):
    for filename in os.listdir(file_pathname):
        filename = filename
        print(filename)
        img = cv2.imread(file_pathname + '/' + filename)
        imageR1p0 = cv2.flip(img, -1)
        cv2.imwrite(r"D:\AI_fileCollected\Datasets\Measurement\test_Image_1" + "/" + filename, imageR1p0)


flip_upsidedown(r"D:\AI_fileCollected\Datasets\Measurement\test_Image")

