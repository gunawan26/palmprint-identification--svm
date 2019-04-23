import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics, preprocessing, svm
from sklearn.metrics import accuracy_score


from image_processing import preprocessing_gambar, feature_extraction

import cv2

from joblib import dump, load

val = "3"
PATH = "D:\\project_biometrika\\dataset\\test\\00{}\\00{}_1.bmp".format(val,val)
MODEL_PATH = "D:\\project_biometrika\\model_10.joblib"
TRAIN_PATH = "D:\\project_biometrika\\dataset\\train"


def main():
    cats = [cats for cats in os.listdir(TRAIN_PATH+"\\.")] # class the images by folder 
    img = cv2.imread(PATH,0)
    model_training = load(MODEL_PATH) 

    img_pre = preprocessing_gambar(img);
    img_feature = feature_extraction(img_pre,"laplacian");
    flatten_img = img_feature.flatten()
    print("image chanel {}".format(flatten_img.shape))
    reshape_img = flatten_img.reshape(1,-1)
    X_test = preprocessing.scale( reshape_img )

    y = model_training.predict(reshape_img)



    print(cats[y[0]])
                # flatten_images.append(aft_feature_ext.flatten())    


    pass

if __name__ == "__main__":
    main()

    pass
