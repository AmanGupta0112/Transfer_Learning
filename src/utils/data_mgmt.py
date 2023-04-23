import time
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import models
import numpy as np
import logging

def get_data(validation_datasize):
    mnist = tf.keras.datasets.mnist
    (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
    x_valid, x_train = x_train_full[:validation_datasize] / \
        255, x_train_full[validation_datasize:]/255
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]
    x_test = x_test/255
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def filename_unique(filename):
    filename = time.strftime(f"{filename}_%Y%m%d%H%M%S")
    return filename


def save_model(model, ARTIFACT_DIR, MODEL_DIR, MODEL_NAME):
    model_dir = f"{ARTIFACT_DIR}/{MODEL_DIR}"
    os.makedirs(model_dir, exist_ok=True)

    # file_name = filename_unique(MODEL_NAME)
    filepath = os.path.join(model_dir, f"{MODEL_NAME}.h5")
    model.save(filepath)
    return f"{MODEL_NAME}.h5"


def plot_data(history, ARTIFACT_DIR, PLOT_DIR, PLOT_NAME):
    df = pd.DataFrame(history.history)
    plot_dir = f"{ARTIFACT_DIR}/{PLOT_DIR}"
    os.makedirs(plot_dir, exist_ok=True)  # Only create if dir doesn't exist
    file_name = filename_unique(PLOT_NAME)
    filepath = os.path.join(plot_dir, f'{file_name}.png')
    df.plot(figsize=(10, 7))
    plt.grid(True)
    plt.savefig(filepath)


def predict(ARTIFACT_DIR, MODEL_DIR, MODEL_NAME, PLOT_DIR, PREDICTION_IMAGE, x_test, y_test):
    model = models.load_model(
        f"{ARTIFACT_DIR}/{MODEL_DIR}/{MODEL_NAME}")
    logging.info(f"Model evaluation : {model.evaluate(x_test, y_test)}")
    x_new = x_test[:30]
    y_prob = model.predict(x_new)
    y_prob.round(3)
    y_pred = np.argmax(y_prob, axis=-1)
    counter = 0
    for img_array, pred, actual in zip(x_new, y_pred, y_test[:30]):
        os.makedirs(
            f"{ARTIFACT_DIR}/{PLOT_DIR}/{PREDICTION_IMAGE}", exist_ok=True)
        plt.imshow(img_array, cmap="binary")
        plt.title(f'predicted: {pred}, Actual: {actual}')
        plt.axis('off')
        plt.savefig(
            f"{ARTIFACT_DIR}/{PLOT_DIR}/{PREDICTION_IMAGE}/Image{counter}.png")
        counter += 1


def get_log_path(Log_dir="logs/fit"):
    uniqueName = time.strftime("log_%Y_%m_%d_%H_%M_%S")
    log_path = os.path.join(Log_dir, uniqueName)

    return log_path


def create_log(log_dir, x_train):
    log_dir = get_log_path()
    file_writer = tf.summary.create_file_writer(logdir=log_dir)
    with file_writer.as_default():
        images = np.reshape(x_train[10:30], (-1, 28, 28, 1))
        tf.summary.image("20 hand written digit sample",
                         images, max_outputs=25, step=0)

def load_models(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def callback_function(log_dir, ARTIFACT_DIR, CKPT_MODEL):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True)
    CKPT_path = f"{ARTIFACT_DIR}/checkpoint/{CKPT_MODEL}.h5"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        CKPT_path, save_best_only=True)
    return [tensorboard_callback, early_stopping_cb, checkpoint_cb]


def train_model_checkpoint(ARTIFACT_DIR, CKPT_MODEL, EPOCHS, x_train, y_train, VALIDATION, CallBack_list):
    CKPT_path = f"{ARTIFACT_DIR}/checkpoint/{CKPT_MODEL}.h5"
    ckpt_model = load_models(CKPT_path)
    history = ckpt_model.fit(x_train, y_train, epochs=EPOCHS,
                             validation_data=VALIDATION, callbacks=CallBack_list)
    return history


def base_model_update(model):
    base_model = model
    for layer in base_model.layers[:-1]:
        print(f"Trainable status of Before {layer.name} : {layer.trainable}")
        layer.trainable = False


def base_model_info(model):
    base_model = model
    for layer in base_model.layers:
        print(f"Trainable status of After {layer.name} : {layer.trainable}")

