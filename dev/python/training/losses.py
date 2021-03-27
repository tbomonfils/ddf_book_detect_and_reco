import tensorflow as tf


def MSE_MSLE(y_true, y_pred):
    MSE = tf.keras.losses.MSE(label_array[:, :8], model(img_batch)[:, :8])
    MSLE = tf.keras.losses.MSLE(label_array[:, -1], model(img_batch)[:, -1])
    return MSE+MSLE