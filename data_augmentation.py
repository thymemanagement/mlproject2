import cv2
from numpy import asarray, save, load
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import os

from preprocess import parse_training_range

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0,1.0))),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-30,30),
        order=[0,1],
        mode=ia.ALL
    ),
    iaa.MultiplyAndAddToBrightness(mul=(0.5,1.0), add=(-30, 30))
], random_order=True
)

def augment_data(images, labels, pipeline, count):
    new_labels = []
    new_images = []
    for i in range(len(images)):
        for n in range(count):
            new_image = pipeline(image=images[i])
            new_labels.append(labels[i])
            new_images.append(new_image)
    return (np.array(new_images), np.array(new_labels))

def main():
    augment_images_path = 'augment_images.npy'
    augment_labels_path = 'augment_labels.npy'
    
    csv_path = 'Train_Labels.csv'
    img_path = 'Train_Image/Train_Image/'
    n = 5
    
    s_images, s_labels = parse_training_range(csv_path, img_path, 20, 30)
    as_images, as_labels = augment_data(s_images, s_labels, seq, n)
    for i in range(len(s_images)):
        cv2.imshow(s_labels[i].value, s_images[i])
        cv2.waitKey(0) 
        for j in range(n):
            cv2.imshow(as_labels[n * i + j].value, as_images[n * i + j])
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    

if (__name__ == '__main__'):
    main()
            
