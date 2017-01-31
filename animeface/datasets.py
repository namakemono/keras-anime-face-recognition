#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def get_label_index_mapping():
    return {label: index for index, label in enumerate(os.listdir("../input/"))}

def load_data(img_rows, img_cols):
    X, y = [], []
    label_index_mapping = get_label_index_mapping()
    for fpath in sorted(glob.glob("../output/*/*.png")):
        print fpath
        img = cv2.imread(fpath)
        label = fpath.split("/")[-2]
        X.append(cv2.resize(img, (img_rows, img_cols)))
        y.append(label_index_mapping[label])
    X, y = np.asarray(X, dtype=np.float), np.asarray(y, dtype=np.int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return (X_train, y_train), (X_test, y_test)    

