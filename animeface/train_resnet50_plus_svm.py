#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import os, sys
import glob
import numpy as np
import pandas as pd
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from datasets import get_label_index_mapping

def extract_features():
    X, y = [], []
    label_index_mapping = get_label_index_mapping()
    for fpath in sorted(glob.glob("../output/*/*.png")):
        img = cv2.imread(fpath)
        label = fpath.split("/")[-2]
        X.append(cv2.resize(img, (224, 224)))
        y.append(label_index_mapping[label])
    X, y = np.asarray(X, dtype=np.float), np.asarray(y, dtype=np.int)
    X = preprocess_input(X)
    model = ResNet50(include_top=False)
    feat = model.predict(X, batch_size=4, verbose=1)
    df = pd.DataFrame(feat.reshape(feat.shape[0], feat.shape[-1]))
    df["target"] = y
    df.to_csv("../output/feats.csv", index=False)

def train():
    df = pd.read_csv("../output/feats.csv")
    y = df["target"].values
    del df["target"]
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    parameters = {"C":[1, 10, 100, 1000], "gamma": [1e-2, 1e-3, 1e-4]}
    clf = GridSearchCV(SVC(), parameters, verbose=1)
    clf.fit(X_train, y_train)
    print "Accuracy: ", clf.score(X_test, y_test) 

if __name__ == "__main__":
    extract_features()
    train() 
