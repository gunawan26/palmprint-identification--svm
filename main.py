import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics, preprocessing, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import Bunch
from image_processing import preprocessing_gambar, feature_extraction

import cv2
import pickle
from joblib import dump, load


PATH = "D:\\project_biometrika\\dataset\\train"
WEIGHT, HEIGHT = 150, 150

RZ_WIDTH = 75
RZ_HEIGHT = 75
"""

Proses : Load gambar 
        -> preprocessing(gaussian filter) 
        -> feature_feature extraction(laplacian) 
        -> classifier(SVM)

"""


def load_gambar():
    # class the images by folder
    cats = [cats for cats in os.listdir(PATH+"\\.")]
    deskripsi = "klasifikasi telapak tangan"
    images = []
    flatten_images = []
    target = []

    for i, dir_images in zip(range(10), os.listdir(PATH)):
        for file in os.listdir(PATH+"\\{}".format(dir_images)):
            # get image from folder dir category
            dir = os.path.join(PATH+"\\{}".format(dir_images), file)
            # load image
            img = cv2.imread(dir, 0)
            after_preprocessing_img = preprocessing_gambar(
                img)  # preprocessing image
            # feature extraction laplacian and gabor, default laplacian
            aft_feature_ext = feature_extraction(
                after_preprocessing_img, 'gabor')
        #     print("image chanel {}".format(len(aft_feature_ext.shape)))
            # flatten the images to 1 dimensional numpy array
            flatten_images.append(aft_feature_ext.flatten())
            images.append(aft_feature_ext)  # append numpy array image to list
            target.append(i)
    flatten_images = np.array(flatten_images)
    images = np.array(images)
    target = np.array(target)
    print("sukses")

    return Bunch(data=flatten_images,
                 target=target,
                 target_names=cats,
                 images=images,
                 descr=deskripsi
                 )


def main():
    np.set_printoptions(threshold=sys.maxsize)
    print("mencoba load gambar ...")
    image_dataset = load_gambar()
    print("load gambar sukses...")

    print('melakukan split test')
    X_train, X_test, y_train, y_test = train_test_split(
        image_dataset.data, image_dataset.target, test_size=0.1)

    print('split test selesai')
#     X_train =preprocessing.scale( X_train )
#     X_test = preprocessing.scale( X_test )

    print('mencoba melakukan training')
    estimator = svm.LinearSVC(max_iter=10000)
    param_grid = [
        {'multi_class': ['ovr'], 'C': [1000, 10000], 'dual':[False]},
        {'C': [1000, 10000], 'multi_class': ['crammer_singer']}
    ]
    clf = GridSearchCV(estimator, param_grid=param_grid,
                       cv=5, refit=True, n_jobs=-1)
    clf.fit(X_train, y_train)
    print('training selesai')
    y_pred = clf.predict(X_test)

    print(y_test)
    print(y_pred)
    #print("Classification report for - \n{}:\n{}\n".format(clf, metrics.classification_report(y_test, y_pred)))
    print(accuracy_score(y_test, y_pred))
    # print(X_train[0])
    dump(clf, 'model_10_gabor.joblib')

    pass


if __name__ == "__main__":
    main()
    pass
