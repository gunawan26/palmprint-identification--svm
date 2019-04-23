import os
from pathlib import Path
import numpy as np
import cv2


def preprocessing_gambar(raw_image_args):
    img = cv2.GaussianBlur(raw_image_args, (5, 5), 0)
    #  return cv2.dilate(img, np.ones((5,5), np.uint8) , iterations=1)
    return img


def feature_extraction(after_pre_img_args, ext_feature='laplacian'):

    if ext_feature == 'laplacian':
        val = cv2.Laplacian(after_pre_img_args, cv2.CV_64F)
        # after_threshold = cv2.adaptiveThreshold(val,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
        abs_dst = cv2.convertScaleAbs(val)
        # abs_dst = cv2.adaptiveThreshold(abs_dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        #ret,th2 = cv2.threshold(abs_dst,15,255,cv2.THRESH_BINARY)
        return abs_dst
    elif ext_feature == 'gabor':
        return gabor_filter(after_pre_img_args)


def gabor_filter(image_args):
    gkernel = cv2.getGaborKernel(
        (11, 11), 6.0, np.pi/4, 9.0, 1, 1, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(image_args, cv2.CV_8UC3, gkernel)
    return filtered_img
