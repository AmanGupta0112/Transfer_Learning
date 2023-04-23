import argparse
import os
import shutil
from tqdm import tqdm
import logging
import tensorflow as tf
import numpy as np
from src.utils.common import read_yaml, create_directories
from src.utils.data_mgmt import get_data
from src.utils.models import create_model

STAGE = "Creating Base Model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    params = config['params']
    artifacts = config['artifacts']
    # EPOCHS = config['params']['epochs']
    # MODEL_DIR = config['artifacts']['model_dir']
    # ARTIFACT_DIR = config['artifacts']['artifacts_dir']
    # MODEL_NAME = config['artifacts']['model_name']
    # PLOT_DIR = config['artifacts']['plots_dir']
    # PLOT_NAME = config['artifacts']['plot_name']
    # PREDICTION_IMAGE = config['artifacts']['prediction_image_dir']
    # CKPT_MODEL = config['artifacts']['checkpoint_model']
    (x_train, y_train), (x_valid, y_valid), (x_test,
                                             y_test) = get_data(params['validation_datasize'])
    seed = params['seed']
    # tf.random.set_seed(seed)
    np.random.seed(seed)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    model = create_model(params['loss_function'], optimizer, params['metrics'],
                         params['no_classes'])
    logging.info(f"Model Summary : {model.summary()}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e