from openvino.runtime import Core
import cv2
import numpy as np
from sklearn.isotonic import IsotonicRegression
import os


SIZE = 224
STRIDE = 224


def load_image(im):
    im = im.astype("float32")
    height, width, channels = im.shape
    if height < SIZE or width < SIZE:
        # Calculate the padding needed for each side
        pad_height = max(0, SIZE - height)
        pad_width = max(0, SIZE - width)

        # Top-bottom padding: (pad on top, pad on bottom)
        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad

        # Left-right padding: (pad on left, pad on right)
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad

        # Apply the padding
        im = np.pad(
            im,
            [[top_pad, bottom_pad], [left_pad, right_pad], [0, 0]],
            constant_values=255,
        )
    return im


def iter_strides(shape):
    for y in range(0, shape[0] - SIZE + 1, STRIDE):
        for x in range(0, shape[1] - SIZE + 1, STRIDE):
            yield y, x


def img_to_slices_one_dir(im):
    assert im.shape[0] >= SIZE, im.shape
    assert im.shape[1] >= SIZE, im.shape

    slices = []
    for y, x in iter_strides(im.shape):
        slices.append(im[y : y + SIZE, x : x + SIZE])
    return np.array(slices)


def slices_to_img_one_dir(slices, shape):
    count = np.zeros(shape)
    data = np.zeros(shape)

    for slc, (y, x) in zip(slices, iter_strides(shape)):
        data[y : y + SIZE, x : x + SIZE] += slc
        count[y : y + SIZE, x : x + SIZE] += 1

    return data, count


def img_to_slices(im):
    slices = [
        img_to_slices_one_dir(im),
        img_to_slices_one_dir(im[::-1]),
        img_to_slices_one_dir(im[:, ::-1]),
        img_to_slices_one_dir(im[::-1, ::-1]),
    ]
    return np.concatenate(slices, axis=0)


def slices_to_img(slices, shape):
    shape = shape[:2] + slices.shape[3:]
    count = np.zeros(shape)
    data = np.zeros(shape)

    assert len(slices) % 4 == 0, slices.shape
    slices_per_dir = len(slices) // 4

    d, c = slices_to_img_one_dir(slices[0:slices_per_dir], shape)

    data += d
    count += c

    d, c = slices_to_img_one_dir(slices[slices_per_dir : slices_per_dir * 2], shape)
    data += d[::-1]
    count += c[::-1]

    d, c = slices_to_img_one_dir(slices[slices_per_dir * 2 : slices_per_dir * 3], shape)
    data += d[:, ::-1]
    count += c[:, ::-1]

    d, c = slices_to_img_one_dir(slices[slices_per_dir * 3 :], shape)
    data += d[::-1, ::-1]
    count += c[::-1, ::-1]

    assert np.min(count) > 0, np.min(count)
    return data / count


def sigmoid(x):
    """Applies the sigmoid function."""
    return 1 / (1 + np.exp(-x))


def infer_with_openvino(infer_request, input_data):
    """Runs inference using OpenVINO Runtime."""
    # Get input layer information
    input_tensor_name = compiled_model.input(0).get_any_name()

    # Run inference
    infer_request.infer(inputs={input_tensor_name: input_data})

    # Get the output
    return infer_request.get_output_tensor(0).data


def binarize(gray_img):
    im = load_image(gray_img)
    orig_shape = im.shape
    im = img_to_slices(im)
    for i in range(0, len(im), 64):
        chunk = infer_with_openvino(infer_request, im[i : i + 64])
        sh = chunk.shape
        chunk = sigmoid(chunk.flatten())
        chunk = isoregr.predict(chunk).reshape(sh)
        im[i : i + 64] = chunk
    im = slices_to_img(im, orig_shape)
    im = (np.clip(im, 0.0, 1.0) * 255.0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel for erosion
    img = cv2.dilate(im, kernel, iterations=1)
    return img


core = Core()
model_path = os.path.join(os.path.dirname(__file__), "model/binarizer.xml")
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name="CPU")
infer_request = compiled_model.create_infer_request()
isoregr_params = np.load(os.path.join(os.path.dirname(__file__), "model/isoregr.npz"))
isoregr = IsotonicRegression().fit(isoregr_params["X"], isoregr_params["y"])
