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
    iaa.SomeOf((1,1),
        [
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-10,10),
                order=[0,1],
                mode=ia.ALL
            ),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                rotate=(-30,30),
                order=[0,1],
                mode=ia.ALL
            ),
        ], random_order=True
    ),
    iaa.SomeOf((1,3),
        [
            iaa.MultiplyAndAddToBrightness(mul=(0.5,1.0), add=(-30, 30)),
            iaa.MultiplySaturation((0.5,1.5)),
            iaa.GammaContrast((0.5, 1.5)),
            iaa.LinearContrast((0.8,1.6)),
            iaa.Sometimes(0.1, iaa.Snowflakes(flake_size=(0.1,0.3), speed=(0.01, 0.05))),
            iaa.ChangeColorTemperature((1100,13000)),
            iaa.SigmoidContrast(gain=(3,10), cutoff=(0.4, 0.6))
        ], random_order=True)
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
    csv_path = 'Train_Labels.csv'
    img_path = 'Train_Image/Train_Image/'
    n = 5
    
    s_images, s_labels = parse_training_range(csv_path, img_path, 500, 510)
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
            
