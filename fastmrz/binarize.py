import cv2
import itertools
import numpy as np
import os

net = cv2.dnn.readNetFromONNX(
    os.path.join(os.path.dirname(__file__), "model/de_gan_fp16.onnx")
)


def split2(dataset, size, h, w):
    newdataset = []
    nsize1 = 256
    nsize2 = 256
    for i in range(size):
        im = dataset[i]
        for ii, iii in itertools.product(range(0, h, nsize1), range(0, w, nsize2)):
            newdataset.append(im[ii : ii + nsize1, iii : iii + nsize2, :])
    return np.array(newdataset)


def merge_image2(splitted_images, h, w):
    image = np.zeros(((h, w, 1)))
    nsize1 = 256
    nsize2 = 256
    ind = 0
    for ii in range(0, h, nsize1):
        for iii in range(0, w, nsize2):
            image[ii : ii + nsize1, iii : iii + nsize2, :] = splitted_images[ind]
            ind = ind + 1
    return np.array(image)


def binarize(img):
    img = img.astype("float32") / 255.0

    # Prepare padding
    h = ((img.shape[0] // 256) + 1) * 256
    w = ((img.shape[1] // 256) + 1) * 256
    test_padding = np.ones((h, w))
    test_padding[: img.shape[0], : img.shape[1]] = img
    test_image_p = split2(test_padding.reshape(1, h, w, 1), 1, h, w)
    predicted_list = []
    for patch in test_image_p:
        net.setInput(patch)
        output = net.forward()
        predicted_list.append(output.reshape(256, 256, 1))

    predicted_image = np.array(predicted_list)
    predicted_image = merge_image2(predicted_image, h, w)

    # Crop the predicted image to original size
    predicted_image = predicted_image[: img.shape[0], : img.shape[1]]

    # Apply binary thresholding
    bin_thresh = 0.95
    return (((predicted_image[:, :, 0] > bin_thresh) * 1) * 255).astype(np.uint8)
