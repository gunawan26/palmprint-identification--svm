import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def main(image_args):
    print("main")

    # dir = os.path.join(PATH,FILE)
    # test_raw_image = cv2.imread(dir,0)
    print(type(image_args))
    # test_image = cv2.GaussianBlur(image_args,(3,3),0) # 7x7 for laplacian # 3x3 for gabor kernel
    th2,th3 = laplacian_filter(image_args)
   
    # cv2.imshow("telapak tangan laplacioan",th2)
    # cv2.imshow("telapak tangan laplacian tanpa thres",th3)
    # th4 = gabor_filter(test_raw_image)
    # cv2.imshow("telapak tangan gabor",th4)
    np_gambar = np.array(th2)   
    print(np_gambar.min())
    print(np_gambar.max())

    
    return th2

def gabor_filter(image_args):
    gkernel =  cv2.getGaborKernel((11, 11), 6.0, np.pi/4, 9.0, 1, 1, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(image_args, cv2.CV_8UC3, gkernel)
    return filtered_img

def laplacian_filter(test_raw_image):
    test_image = cv2.GaussianBlur(test_raw_image,(5,5),0)
    laplacian_output = cv2.Laplacian(test_image,cv2.CV_64F)
    abs_dst_n_thres = cv2.convertScaleAbs(laplacian_output)
    # abs_dst = cv2.adaptiveThreshold(abs_dst_n_thres,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,73,2)
    abs_dst = cv2.adaptiveThreshold(abs_dst_n_thres,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #ret,th2 = cv2.threshold(abs_dst,7,255,cv2.THRESH_BINARY) bad result
    return abs_dst,abs_dst_n_thres

if __name__ == "__main__":

    main()
    print("process end ...")
    pass