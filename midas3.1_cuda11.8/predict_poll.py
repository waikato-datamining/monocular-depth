import os
import argparse
from image_complete import auto
import torch
import traceback

from sfp import Poller
from predict_common import prediction_to_file, PREDICTION_FORMATS_FILE, PREDICTION_FORMAT_GRAYSCALE, load_model, load_image_file, MODEL_TYPES, ModelContainer, predict


SUPPORTED_EXTS = [".jpg", ".jpeg"]
""" supported file extensions (lower case). """


def check_image(fname, poller):
    """
    Check method that ensures the image is valid.

    :param fname: the file to check
    :type fname: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: True if complete
    :rtype: bool
    """
    result = auto.is_image_complete(fname)
    poller.debug("Image complete:", fname, "->", result)
    return result


def process_image(fname, output_dir, poller):
    """
    Method for processing an image.

    :param fname: the image to process
    :type fname: str
    :param output_dir: the directory to write the image to
    :type output_dir: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: the list of generated output files
    :rtype: list
    """
    result = []

    try:
        cont = poller.params.model_cont
        original_image_rgb = load_image_file(fname)
        image = cont.transform({"image": original_image_rgb})["image"]
        with torch.no_grad():
            pred = predict(poller.params.model_cont, image, original_image_rgb.shape[1::-1])
        fname_out = os.path.join(output_dir, os.path.splitext(os.path.basename(fname))[0] + ".png")
        fname_out = prediction_to_file(pred, poller.params.prediction_format, fname_out)
        result.append(fname_out)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process image: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict_on_images(model_cont, input_dir, output_dir, tmp_dir, prediction_format="grayscale",
                      poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
                      delete_input=False, verbose=False, quiet=False):
    """
    Method for performing predictions on images.

    :param model_cont: the model container with the model
    :type model_cont: ModelContainer
    :param input_dir: the directory with the images
    :type input_dir: str
    :param output_dir: the output directory to move the images to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished, use None if not to use
    :type tmp_dir: str
    :param prediction_format: the format to use for the prediction images (grayscale/bluechannel)
    :type prediction_format: str
    :param poll_wait: the amount of seconds between polls when not in watchdog mode
    :type poll_wait: float
    :param continuous: whether to poll continuously
    :type continuous: bool
    :param use_watchdog: whether to react to file creation events rather than use fixed-interval polling
    :type use_watchdog: bool
    :param watchdog_check_interval: the interval for the watchdog process to check for files that were missed due to potential race conditions
    :type watchdog_check_interval: float
    :param delete_input: whether to delete the input images rather than moving them to the output directory
    :type delete_input: bool
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    """

    poller = Poller()
    poller.input_dir = input_dir
    poller.output_dir = output_dir
    poller.tmp_dir = tmp_dir
    poller.extensions = SUPPORTED_EXTS
    poller.delete_input = delete_input
    poller.progress = not quiet
    poller.verbose = verbose
    poller.check_file = check_image
    poller.process_file = process_image
    poller.poll_wait = poll_wait
    poller.continuous = continuous
    poller.use_watchdog = use_watchdog
    poller.watchdog_check_interval = watchdog_check_interval
    poller.params.model_cont = model_cont
    poller.params.prediction_format = prediction_format
    poller.poll()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiDaS - Prediction", prog="midas_predict_poll", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', help='The device to use (auto|cpu|cuda)', required=False, default="auto")
    parser.add_argument('--model_weights', help='Path to the weights', required=True, default=None)
    parser.add_argument('--model_type', help='The type of model to use: ' + MODEL_TYPES, required=True, default=None)
    parser.add_argument('--optimize', action='store_true', help='Whether to optimize using half-float precision', required=False, default=False)
    parser.add_argument('--height', type=int, help='Preferred height of images feed into the encoder during inference. Note that the preferred height may differ from the actual height, because an alignment to multiples of 32 takes place. Many models support only the height chosen during training, which is used automatically if this parameter is not set.', required=False, default=None)
    parser.add_argument('--square', action='store_true', help='Option to resize images to a square resolution by changing their widths when images are fed into the encoder during inference. If this parameter is not set, the aspect ratio of images is tried to be preserved if supported by the model.', required=False, default=False)
    parser.add_argument('--prediction_in', help='Path to the test images', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output csv files folder', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary csv files folder', required=False, default=None)
    parser.add_argument('--prediction_format', default=PREDICTION_FORMAT_GRAYSCALE, choices=PREDICTION_FORMATS_FILE, help='The format for the prediction images')
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args()

    try:
        model_cont = load_model(parsed.device, parsed.model_weights, parsed.model_type, parsed.optimize, parsed.height, parsed.square)

        # Performing the prediction and producing the predictions files
        predict_on_images(model_cont, parsed.prediction_in, parsed.prediction_out, parsed.prediction_tmp,
                          prediction_format=parsed.prediction_format, continuous=parsed.continuous,
                          use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
                          delete_input=parsed.delete_input, verbose=parsed.verbose, quiet=parsed.quiet)

    except Exception as e:
        print(traceback.format_exc())
