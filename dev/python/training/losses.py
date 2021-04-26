import tensorflow as tf


def MSE_MSLE(y_true, y_pred):
	MSE = tf.reduce_mean(tf.keras.losses.MSE(y_true[:, :8], y_pred[:, :8]))
	MSLE = tf.keras.losses.MSLE(y_true[:, -1], y_pred[:, -1])
	return MSE+MSLE