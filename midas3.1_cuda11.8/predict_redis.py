from datetime import datetime
import numpy as np
import torch
import traceback

from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from predict_common import prediction_to_data, PREDICTION_FORMATS_DATA, PREDICTION_FORMAT_GRAYSCALE, load_model, load_image_array, MODEL_TYPES, ModelContainer, predict


def process_image(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    model_cont = msg_cont.params.model_cont

    try:
        start_time = datetime.now()

        array = np.frombuffer(msg_cont.message['data'], np.uint8)
        original_image_rgb = load_image_array(array)
        image = model_cont.transform({"image": original_image_rgb})["image"]
        with torch.no_grad():
            pred = predict(model_cont, image, original_image_rgb.shape[1::-1])
        out_data = prediction_to_data(pred, msg_cont.params.prediction_format)
        msg_cont.params.redis.publish(msg_cont.params.channel_out, out_data)

        if config.verbose:
            log("process_images - prediction image published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_images - finished processing image: %d ms" % processing_time)

    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_images - failed to process: %s" % traceback.format_exc())


if __name__ == '__main__':
    parser = create_parser('MiDaS - Prediction (Redis)', prog="midas_predict_redis", prefix="redis_")
    parser.add_argument('--device', help='The device to use (auto|cpu|cuda)', required=False, default="auto")
    parser.add_argument('--model_weights', help='Path to the weights', required=True, default=None)
    parser.add_argument('--model_type', help='The type of model to use: ' + MODEL_TYPES, required=True, default=None)
    parser.add_argument('--optimize', action='store_true', help='Whether to optimize using half-float precision', required=False, default=False)
    parser.add_argument('--height', type=int, help='Preferred height of images feed into the encoder during inference. Note that the preferred height may differ from the actual height, because an alignment to multiples of 32 takes place. Many models support only the height chosen during training, which is used automatically if this parameter is not set.', required=False, default=None)
    parser.add_argument('--square', action='store_true', help='Option to resize images to a square resolution by changing their widths when images are fed into the encoder during inference. If this parameter is not set, the aspect ratio of images is tried to be preserved if supported by the model.', required=False, default=False)
    parser.add_argument('--prediction_format', default=PREDICTION_FORMAT_GRAYSCALE, choices=PREDICTION_FORMATS_DATA, help='The format for the prediction images')
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parsed = parser.parse_args()

    try:
        model_cont = load_model(parsed.device, parsed.model_weights, parsed.model_type, parsed.optimize, parsed.height, parsed.square)

        config = Container()
        config.model_cont = model_cont
        config.prediction_format = parsed.prediction_format
        config.verbose = parsed.verbose

        params = configure_redis(parsed, config=config)
        run_harness(params, process_image)

    except Exception as e:
        print(traceback.format_exc())
