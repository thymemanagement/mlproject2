import cv2
from numpy import asarray, save, load
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0,3.0))),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-30,30),
        order=[0,1],
        mode=ia.ALL
    ),
    iaa.MultiplyAndAddToBrightness(mul=(0.5,1.5), add=(-30, 30))
], random_order=True
)

def augment_data(labels, images, pipeline, count, debug=False):
    new_labels = []
    new_images = []
    for i in range(len(images)):
        new_labels.append(labels[i])
        new_images.append(images[i])
        if debug:
            cv2.imshow(images[i])
        for n in range(count):
            new_image = pipeline.augment_image(images[i])
            new_labels.append(labels[i])
            new_images.append(new_image)
            if debug:
                cv2.imshow(new_image)
    return (np.array(new_labels), np.array(new_images))

def main():
    augment_images_path = 'augment_images.npy'
    augment_labels_path = 'augment_labels.npy'

if (__name__ == '__main__'):
    main()
            
