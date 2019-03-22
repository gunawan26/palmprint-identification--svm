import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics, svm, preprocessing 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import Bunch

import cv2

#from skimage.io import imread
#from skimage.transform import resize
# from skimage.io import imread

PATH = "D:\\project_biometrika\\dataset\\train"
WEIGHT,HEIGHT = 150,150

RZ_WIDTH = 75
RZ_HEIGHT = 75
def load_gambar():

    class_folders = [class_dirs for class_dirs in os.listdir(PATH) if os.path.isdir(PATH+"\\{}".format(class_dirs))]
    cats = [cats for cats in os.listdir(PATH+"\\.")]
    
    deskripsi = "klasifikasi telapak tangan"

    images = []
    flatten_images = []
    target = []

    print(len(os.listdir(PATH)))

    for i, dir_images in zip(range(30),os.listdir(PATH)):
        for file in os.listdir(PATH+"\\{}".format(dir_images)):
            dir = os.path.join(PATH+"\\{}".format(dir_images),file)
            #get images
            img = cv2.imread(dir,0)
            #preprocessing process
            # img = cv2.resize(img,(int(RZ_WIDTH),int(RZ_HEIGHT)))
            after_preprocessing_img = preprocessing_gambar(img)
            #after feature ext 
            aft_feature_ext = feature_extraction(after_preprocessing_img)
            #flatten the images
            flatten_images.append(aft_feature_ext.flatten())
            #append images
            images.append(aft_feature_ext)
            target.append(i)

    flatten_images = np.array(flatten_images)
    images = np.array(images)
    target = np.array(target)



            #flatten_images.append(cv2.imread(dir,0))
    print("sukses")

    return Bunch(data = flatten_images,
                target = target,
                target_names = cats,
                images = images,
                descr = deskripsi
                )
    pass


def preprocessing_gambar(raw_image_args):
    img = cv2.GaussianBlur(raw_image_args,(5,5),0)
  #  return cv2.dilate(img, np.ones((5,5), np.uint8) , iterations=1) 
    return img


def feature_extraction(after_pre_img_args,ext_feature = 'laplacian'):
    
    if ext_feature == 'laplacian':
        val = cv2.Laplacian(after_pre_img_args,cv2.CV_64F)
        abs_dst = cv2.convertScaleAbs(val)
        # abs_dst = cv2.adaptiveThreshold(abs_dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)

        #ret,th2 = cv2.threshold(abs_dst,15,255,cv2.THRESH_BINARY)
        return abs_dst
    elif ext_feature == 'gabor':
        return gabor_filter(after_pre_img_args)
   


def gabor_filter(image_args):
    gkernel =  cv2.getGaborKernel((15, 15), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(image_args, cv2.CV_8UC3, gkernel)
    return filtered_img



def main():
    np.set_printoptions(threshold=sys.maxsize)
    print("mencoba load gambar ...")
    image_dataset = load_gambar()
    print("load gambar sukses...")

    print('melakukan split test')

    # print(image_dataset.images)

    X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.1)
    print('split test selesai')
    # scaler = StandardScaler()
    X_train =preprocessing.scale( X_train )
    X_test = preprocessing.scale( X_test )

    """
    param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    """
    print('mencoba melakukan training')
    clf = svm.LinearSVC(multi_class='crammer_singer',max_iter=5000)
   # clf = GridSearchCV(svc, param_grid)

    clf.fit(X_train, y_train)

    print('training selesai')

    y_pred = clf.predict(X_test)    

    #print("Classification report for - \n{}:\n{}\n".format(clf, metrics.classification_report(y_test, y_pred)))
    print(accuracy_score(y_test, y_pred))
    # print(X_train[0])

    pass

if __name__ == "__main__":
    main()
    pass
