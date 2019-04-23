import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

PATH = "D:\\project_biometrika\\dataset\\train\\007"
FILE = "007_6.bmp"


"""
gaussian filter for gabor 7x7
gaussian filter for laplacian 5x5

"""


def main():
    print("main")

    dir = os.path.join(PATH, FILE)
    test_raw_image = cv2.imread(dir, 0)
    new_preprocessing = cv2.imread(dir, 0)
    new_preprocessing = preprocessing_feat_ex(new_preprocessing)
    # test_image = cv2.GaussianBlur(test_raw_image, (3, 3), 0)
    # 7x7 for laplacian # 3x3 for gabor kernel
    th2, th3 = laplacian_filter(test_raw_image)
    imgplot = plt.imshow(new_preprocessing, cmap="gray")
    plt.show(imgplot)
    # cv2.imshow("new",new_preprocessing)
    # cv2.imshow("blur",test_image)
    # cv2.imshow("telapak tangan laplacioan",th2)
    # cv2.imshow("telapak tangan laplacian tanpa thres",th3)
    th4 = gabor_filter(test_raw_image)
    # cv2.imshow("telapak tangan gabor",th4)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pass


def gabor_filter(image_args, kernel_args=3):
    image_args = cv2.GaussianBlur(image_args, (kernel_args, kernel_args), 0)
    gkernel = cv2.getGaborKernel(
        (11, 11), 6.0, np.pi/4, 9.0, 1, 1, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(image_args, cv2.CV_8UC3, gkernel)
    return filtered_img


def laplacian_filter(test_raw_image):
    test_image = cv2.GaussianBlur(test_raw_image, (5, 5), 0)
    laplacian_output = cv2.Laplacian(test_raw_image, cv2.CV_64F)
    abs_dst_n_thres = cv2.convertScaleAbs(laplacian_output)
    # abs_dst = cv2.adaptiveThreshold(abs_dst_n_thres,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,73,2)
    abs_dst = cv2.adaptiveThreshold(
        abs_dst_n_thres, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # ret,th2 = cv2.threshold(abs_dst,7,255,cv2.THRESH_BINARY) bad result
    return abs_dst, abs_dst_n_thres


def preprocessing_feat_ex(img_args):
    # test_image = cv2.GaussianBlur(img_args,(3,3),0)
    equ = cv2.equalizeHist(img_args)
    res = np.hstack((img_args, equ))
    res = gabor_filter(res)
    return res


if __name__ == "__main__":

    main()
    print("process end ...")
    pass
