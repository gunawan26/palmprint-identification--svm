import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

PATH = "D:\\project_biometrika\\dataset\\train\\007"
FILE = "007_6.bmp"


def main():
    print("main")

    dir = os.path.join(PATH,FILE)
    test_raw_image = cv2.imread(dir,0)
   
    test_image = cv2.GaussianBlur(test_raw_image,(7,7),0)
    #cv2.imshow("telapak tangan",test_image)
    #th3  = cv2.Canny(test_image,50,200)
    #laplacian_output = cv2.Laplacian(test_image,cv2.CV_64F)
    #abs_dst = cv2.convertScaleAbs(laplacian_output)
    #th2 = cv2.adaptiveThreshold(abs_dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #ret,th2 = cv2.threshold(abs_dst,15,255,cv2.THRESH_BINARY)
    #th5  = cv2.Sobel(test_image,cv2.CV_64F,1,0,ksize=5)
    #gkernel =  cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

    #filtered_img = cv2.filter2D(test_image, cv2.CV_8UC3, gkernel)
    # th3 = cv2.adaptiveThreshold(test_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #cv2.imshow("telapak tangan canny",th3)
    th2 = laplacian_filter(test_image)
    th2 = cv2.adaptiveThreshold(test_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
    cv2.imshow("telapak tangan laplacioan",th2)
   # cv2.imshow("telapak tangan sobel",th5)
   # cv2.imshow("telapak tangan gabor",filtered_img)
    np_gambar = np.array(th2)
    print(np_gambar.min())
    print(np_gambar.max())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass

def gabor_filter(image_args):
    gkernel =  cv2.getGaborKernel((15, 15), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(image_args, cv2.CV_8UC3, gkernel)
    return filtered_img

def laplacian_filter(test_raw_image):
    test_image = cv2.GaussianBlur(test_raw_image,(5,5),0)
    # test_image = cv2.dilate(test_image, np.ones((5,5), np.uint8) , iterations=1) 
    # equ = cv2.equalizeHist(test_image)
    #cv2.imshow("telapak tangan",test_image)
    #th3  = cv2.Canny(test_image,50,200)
    laplacian_output = cv2.Laplacian(test_image,cv2.CV_64F)
    abs_dst = cv2.convertScaleAbs(laplacian_output)
    #th2 = cv2.adaptiveThreshold(abs_dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
   
    #ret,th2 = cv2.threshold(abs_dst,7,255,cv2.THRESH_BINARY) bad result
    return abs_dst

if __name__ == "__main__":

    main()
    print("process end ...")
    pass