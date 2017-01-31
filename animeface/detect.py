import cv2
import os, sys
import glob

def detect(filename, cascade_file = "lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)
    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (24, 24))
    return image, faces

def run():
    cnt = 0
    for fpath in glob.glob("../input/*/*"):
        print fpath
        image, faces = detect(fpath)
        for x, y, w, h in faces:
            img = image[y:y+h, x:x+w]
            label = fpath.split("/")[-2]
            save_dir = "../output/%s" % label
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite("%s/%d.png" % (save_dir, cnt), img)
            cnt += 1

if __name__ == "__main__":
    run()

