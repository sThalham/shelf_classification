
# Uses model with specified weights to classify images in impath and saves results in classification_results.txt

import tensorflow.keras as K
import numpy as np
import os
import cv2


# The provided models are trained for images with 168x224 resolution
# If a model is used that was trained with segmentation augmentation, it should also be applied to images for
# classification beforehand for the best results

# Path of images to classify
impath = "./images"


# Specify model weights to use
model_weights_path = "./model_weights/segment_and_aug_no_mixup.h5"

classifier = K.models.load_model(model_weights_path)
CLASSES = ["bucket", "hanging", "standing"]

results_file = open("classification_results.txt", "w")
results_file.write("Filename\t Predicted class \t Percentages for bucket, hanging and standing\n")

for filename in os.listdir(impath):
    img = cv2.imread(os.path.join(impath, filename))
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        class_result = classifier.predict(img_rgb[np.newaxis, ...])
        prediction = np.argmax(class_result)
        results_file.write(filename + "     " + str(CLASSES[prediction]) + "     " + str(class_result) + "\n")

results_file.close()





