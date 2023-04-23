import argparse
import os
import shutil
from tqdm import tqdm
import logging
import tensorflow as tf
import numpy as np
import io
from src.utils.common import read_yaml, create_directories
from src.utils.data_mgmt import (get_data, save_model,
                                plot_data,
                                predict,
                                get_log_path,
                                create_log,
                                callback_function,
                                train_model_checkpoint)
from src.utils.models import create_model

STAGE = "Creating Base Model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    config = read_yaml(config_path)
    params = config['params']
    artifacts = config['artifacts']
    (x_train, y_train), (x_valid, y_valid), (x_test,
                                             y_test) = get_data(params['validation_datasize'])
    seed = params['seed']
    # tf.random.set_seed(seed)
    np.random.seed(seed)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    model = create_model(params['loss_function'], optimizer, params['metrics'],
                         params['no_classes'])
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn=lambda x: stream.write(x + "\n"))
            return stream.getvalue()
    logging.info(f"Base model summary:{_log_model_summary(model)}")
    VALIDATION = (x_valid, y_valid)
    log_dir = get_log_path()
    create_log(log_dir, x_train)
    CallBack_list = callback_function(
        log_dir, artifacts['artifacts_dir'], artifacts['checkpoint_model'])
    history = model.fit(x_train, y_train, epochs=params['epochs'],validation_data = VALIDATION, callbacks = CallBack_list)
    file_name = save_model(
        model, artifacts['artifacts_dir'], artifacts['model_dir'], artifacts['model_name'])
    logging.info(f"Base Model File is saved: {file_name}")
    plot_data(history, artifacts['artifacts_dir'], artifacts['plots_dir'], artifacts['plot_name'])
    predict(artifacts['artifacts_dir'], artifacts['model_dir'], file_name,
            artifacts['plots_dir'], artifacts['prediction_image_dir'], x_test, y_test)



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