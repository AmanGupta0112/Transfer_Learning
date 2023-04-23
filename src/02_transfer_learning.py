import argparse
import os,io
import shutil
from tqdm import tqdm
import logging
import tensorflow as tf
import numpy as np
from src.utils.common import read_yaml, create_directories
from src.utils.data_mgmt import (get_data, save_model,
                                 plot_data,
                                 predict,
                                 get_log_path,
                                 create_log,
                                 callback_function,
                                 load_models,
                                 base_model_update,
                                 base_model_info,
                                 train_model_checkpoint)
from src.utils.models import create_model,recreate_model

STAGE = "Transfer Learning"  # <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)


def update_even_odd_labels(list_of_labels):

    for idx, label in enumerate(list_of_labels):
        even_condition = label % 2 == 0
        list_of_labels[idx] = np.where(even_condition, 1, 0)
    return list_of_labels

def main(config_path):
    config = read_yaml(config_path)
    params = config['params']
    artifacts = config['artifacts']
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

    (x_train, y_train), (x_valid, y_valid), (x_test,
                                             y_test) = get_data(params['validation_datasize'])
    
    y_train_binary, y_test_binary, y_valid_binary = update_even_odd_labels(
                                    [y_train, y_test, y_valid])
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn=lambda x: stream.write(x + "\n"))
            return stream.getvalue()
        
    seed = params['seed']
    # tf.random.set_seed(seed)
    np.random.seed(seed)
    base_model_path = os.path.join(
        artifacts['artifacts_dir'], artifacts['model_dir'], f"{artifacts['model_name']}.h5")
    base_model = load_models(base_model_path)

    logging.info(f"Loaded base model summary:{_log_model_summary(base_model)}")
    base_model_update(base_model)
    new_model = recreate_model(base_model, params['loss_function'], optimizer, params['metrics'],
                                                        params['no_classes'])
    
    logging.info(f"Loaded base model summary:{_log_model_summary(new_model)}")
    base_model_info(new_model)
    
    VALIDATION = (x_valid, y_valid_binary)
    log_dir = get_log_path()
    create_log(log_dir, x_train)
    CallBack_list = callback_function(
        log_dir, artifacts['artifacts_dir'], artifacts['checkpoint_model'])
    history = new_model.fit(
        x_train, y_train_binary, epochs=params['epochs'], validation_data=VALIDATION, callbacks=CallBack_list)
    file_name = save_model(
        new_model, artifacts['artifacts_dir'], artifacts['model_dir'], artifacts['new_model_name'])
    logging.info(f"New Model File is saved: {file_name}")
    plot_data(history, artifacts['artifacts_dir'],
              artifacts['plots_dir'], artifacts['new_plot_name'])
    predict(artifacts['artifacts_dir'], artifacts['model_dir'], file_name,
            artifacts['plots_dir'], artifacts['prediction_image_dir'], x_test, y_test_binary)


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
