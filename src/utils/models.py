import tensorflow as tf


def create_model(LOSS_FN, OPTIMIZER, METRICS, NUM_CLASSES):
    LAYERS = [
        tf.keras.layers.Flatten(input_shape=[28, 28], name="inputlayer"),
        # tf.keras.layers.Dense(300, activation='relu', name="hiddenlayer1"),
        # tf.keras.layers.Dense(100, activation='relu', name="hiddenlayer2"),
        tf.keras.layers.Dense(300, name="hiddenlayer1"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(100, name="hiddenlayer2"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name="output")
    ]
    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.compile(loss=LOSS_FN, optimizer=OPTIMIZER, metrics=METRICS)
    return model_clf


def recreate_model(model, LOSS_FN, OPTIMIZER, METRICS, NUM_CLASSES):
    LAYERS = model.layers[:-1]
    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.add(tf.keras.layers.Dense(2, activation='softmax', name="output"))
    model_clf.compile(loss=LOSS_FN, optimizer=OPTIMIZER, metrics=METRICS)
    return model_clf
