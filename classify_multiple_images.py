from shutil import copyfile
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
from imutils import paths
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to folder containing images for classification")
ap.add_argument("-m", "--model", required=True, help="path to model")
args = vars(ap.parse_args())

copy_folder = sorted(list(paths.list_images(args["dataset"])))
paste_folder_elliptical = "test_images/elliptical/"
paste_folder_irregular = "test_images/irregular/"
paste_folder_spiral = "test_images/spiral/"
paste_folder_not_galaxy = "test_images/not_galaxy/"
paste_path = ""

for file in copy_folder :
    image = cv2.imread(file)
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    model = load_model(args["model"])
    (notGalaxy, elliptical, irregular, spiral) = model.predict(image)[0]
    if elliptical > irregular and elliptical > spiral and elliptical > notGalaxy:
        paste_path = paste_folder_elliptical + file.split(os.path.sep)[-1]
    elif irregular > elliptical and irregular > spiral and irregular > notGalaxy:
        paste_path = paste_folder_irregular + file.split(os.path.sep)[-1]
    elif spiral > elliptical and spiral > irregular and spiral > notGalaxy:
        paste_path = paste_folder_spiral + file.split(os.path.sep)[-1]
    else:
        paste_path = paste_folder_not_galaxy + file.split(os.path.sep)[-1]
    copyfile(file, paste_path)	