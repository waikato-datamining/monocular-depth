import cv2
import torch
import numpy as np
import os

from midas.model_loader import load_model as load_model_midas
from pypfm import PFMLoader

PREDICTION_FORMAT_GRAYSCALE = "grayscale"
PREDICTION_FORMAT_GRAYSCALE_DEPTH = "grayscale-depth"
PREDICTION_FORMAT_NUMPY = "numpy"
PREDICTION_FORMAT_PFM = "pfm"
PREDICTION_FORMATS_FILE = [
    PREDICTION_FORMAT_GRAYSCALE,
    PREDICTION_FORMAT_GRAYSCALE_DEPTH,
    PREDICTION_FORMAT_NUMPY,
    PREDICTION_FORMAT_PFM,
]
PREDICTION_FORMATS_DATA = [
    PREDICTION_FORMAT_GRAYSCALE,
    PREDICTION_FORMAT_GRAYSCALE_DEPTH,
    PREDICTION_FORMAT_NUMPY,
]


MODEL_TYPES = \
    'dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, ' \
    'dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, ' \
    'dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256 or ' \
    'openvino_midas_v21_small_256'


class ModelContainer:
    first_execution: bool = False
    device = None
    model_path: str = None
    model_type: str = None
    optimize: bool = False
    height: int = None
    square: bool = False
    model = None
    transform = None
    net_w: int = None
    net_h: int = None



def load_model(device: str, model_path: str, model_type: str, optimize: bool, height: int, square: bool) -> ModelContainer:
    """
    Loads the model.

    :param device: the device to run the model on (auto|cpu|cuda)
    :type device: str
    :param model_path: the path to the weights, can be None
    :type model_path: str
    :param model_type: the type of model to use
    :type model_type: str
    :param optimize: whether to use half-float optimization or not
    :type optimize: bool
    :param height: the preferred height, gets aligned to multiples of 32
    :type height: int
    :param square: whether to resize images to square resolution
    :type square: bool
    :return: the generated container
    :rtype: ModelContainer
    """
    result = ModelContainer()
    result.model_path = model_path
    result.model_type = model_type
    result.optimize = optimize
    result.height = height
    result.square = square
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    result.device = torch.device(device)
    result.model, result.transform, result.net_w, result.net_h = load_model_midas(
        result.device, result.model_path, result.model_type, result.optimize, result.height, result.square)
    return result


def load_image_file(path: str):
    """
    Loads the image from the specified path.

    :param path: the path to load the image from
    :type path: str
    :return: the image
    """
    img = cv2.imread(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img


def load_image_array(array: np.ndarray):
    """
    Loads the image from a numpy array.

    :param array: the numpy array to load the image from
    :type array: np.ndarray
    :return: the image
    """
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img


def predict(cont, image, target_size):
    """
    Run the inference and interpolate.

    :param cont: the model container to use
    :type cont: ModelContainer
    :param image: the image to process
    :param target_size: the size (width, height) the neural network output is interpolated to
    :type target_size: tuple
    :return: the prediction
    """
    input_size = (cont.net_w, cont.net_h)
    if "openvino" in cont.model_type:
        if cont.first_execution:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            cont.first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = cont.model(sample)[cont.model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(cont.device).unsqueeze(0)

        if cont.optimize and cont.device == torch.device("cuda"):
            if cont.first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if cont.first_execution:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            cont.first_execution = False

        prediction = cont.model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def prediction_to_file(depth, prediction_format: str, path: str) -> str:
    """
    Saves the prediction to disk as image using the specified image format.

    :param depth: the midas depth prediction
    :param prediction_format: the image format to use
    :type prediction_format: str
    :param path: the path to save the image to
    :type path: str
    :return: the filename the predictions were saved under
    :rtype: str
    """
    if prediction_format not in PREDICTION_FORMATS_FILE:
        raise Exception("Unsupported format: %s" % prediction_format)

    if prediction_format == PREDICTION_FORMAT_GRAYSCALE:
        bits = 1
    else:
        bits = 2

    if prediction_format in [PREDICTION_FORMAT_GRAYSCALE, PREDICTION_FORMAT_GRAYSCALE_DEPTH]:
        if not np.isfinite(depth).all():
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            print("WARNING: Non-finite depth values present")
        depth_min = depth.min()
        depth_max = depth.max()
        max_val = (2**(8*bits))-1
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)
        if prediction_format == PREDICTION_FORMAT_GRAYSCALE:
            cv2.imwrite(path, out.astype("uint8"))
        elif prediction_format == PREDICTION_FORMAT_GRAYSCALE_DEPTH:
            cv2.imwrite(path + ".png", out.astype("uint16"))
        else:
            raise Exception("Unsupported format: %s" % prediction_format)
    elif prediction_format == PREDICTION_FORMAT_PFM:
        path = os.path.splitext(path)[0] + ".pfm"
        loader = PFMLoader(color=False, compress=False)
        loader.save_pfm(path, depth)
    else:
        path = os.path.splitext(path)[0] + ".npy"
        np.save(path, depth)

    return path


def prediction_to_data(depth, prediction_format: str) -> bytes:
    """
    Saves the prediction to disk as image using the specified image format.

    :param depth: the midas depth prediction
    :param prediction_format: the image format to use
    :type prediction_format: str
    :return: the generated data
    :rtype: bytes
    """
    if prediction_format not in PREDICTION_FORMATS_DATA:
        raise Exception("Unsupported format: %s" % prediction_format)

    if prediction_format == PREDICTION_FORMAT_GRAYSCALE:
        bits = 1
    else:
        bits = 2

    if prediction_format in [PREDICTION_FORMAT_GRAYSCALE, PREDICTION_FORMAT_GRAYSCALE_DEPTH]:
        if not np.isfinite(depth).all():
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            print("WARNING: Non-finite depth values present")
        depth_min = depth.min()
        depth_max = depth.max()
        max_val = (2 ** (8 * bits)) - 1
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)
        if prediction_format == PREDICTION_FORMAT_GRAYSCALE:
            result = cv2.imencode('.png', out.astype("uint8"))[1].tobytes()
        elif prediction_format == PREDICTION_FORMAT_GRAYSCALE_DEPTH:
            result = cv2.imencode('.png', out.astype("uint16"))[1].tobytes()
        else:
            raise Exception("Unsupported format: %s" % prediction_format)
    else:
        result = depth.tobytes()

    return result
