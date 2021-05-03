import os
import shutil
import argparse

import numpy as np
import tensorflow as tf
import random

import PIL.Image as Image

	
#def loss(model, l, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
#    y_ = model(x, training=training)

#    return l(y_true=y, y_pred=y_)


def grad(model, loss, inputs, targets, weight_decay, training):
	with tf.GradientTape() as tape:
		y_ = model(inputs, training=training)
		loss_value = loss(y_true=targets, y_pred=y_)
		loss_value = loss_value + weight_decay*tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if 'batch_normalization' not in v.name])
	return loss_value, tape.gradient(loss_value, model.trainable_variables)

	
def split_train_test_index(index_size, breakdown_ratio=0.8):
	index_split = int(index_size*breakdown_ratio)+1

	index_array = np.array([i for i in range(index_size)])
	random.shuffle(index_array)
	
	index_train, index_test = index_array[:index_split], index_array[index_split:]
	
	return index_train, index_test
	
def batch_img_gen(filenames, augmented_input_dir, real_input_dir, real_data_ratio=0.80 , scale=True):
	img_batch = []
	label_batch = []
	
	for f in filenames:
		if real_data_ratio<random.uniform(0, 1):
			f = os.listdir(real_input_dir)[random.randint(0, len(os.listdir(real_input_dir))-1)]
			img_batch.append(np.array(Image.open(real_input_dir+f)))
		else:
			img_batch.append(np.array(Image.open(augmented_input_dir+f)))

		points = np.array(f.split('.')[0].split('_')[-8:]).reshape((4, 2)).astype(np.int)
		if scale==True:
			label_batch.append(np.concatenate((np.concatenate(np.array(list(zip(points[:,0]/img_batch[-1].shape[-3], \
																points[:,1]/img_batch[-1].shape[-2])))), np.array([1]))))
		else:
			label_batch.append(np.concatenate((points.reshape(8), np.array([1]))))
	
	img_batch_array = np.stack(img_batch)
	label_batch_array = np.stack(label_batch)
	
	return img_batch_array, label_batch_array

def train_model(model, loss, nb_epoch=100, batch_size=64, optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, clipvalue=0.5), weight_decay=5e-4, \
	epoch_target='full', augmented_input_dir='data/output/tests/train_set/', real_input_dir='data/output/tests/annotated_images_train/',
	output_dir='data/model/', breakdown_ratio=0.8, scale_batch_label=True):
	
	file_list = os.listdir(augmented_input_dir)
	nb_obs = len(file_list)
	
	index_train, index_test = split_train_test_index(nb_obs, breakdown_ratio=breakdown_ratio)

	test_index_split = index_test.shape[0]//batch_size
	index_batch_test = index_test[:test_index_split*batch_size].reshape((test_index_split, batch_size))
	
	index_batch_train = index_train[:index_train.shape[0]//batch_size*batch_size].reshape((index_train.shape[0]//batch_size, batch_size))
	
	for epoch in range(nb_epoch):
		epoch_loss_avg = tf.keras.metrics.Mean()
		epoch_accuracy = tf.keras.metrics.BinaryCrossentropy()

		if epoch_target=='full' or index_train.shape[0]<=int(epoch_target):
			filenames_index_gen = (i for i in index_batch_train)
		else:
			batch_indexes = [i for i in range(index_batch_train.shape[0])]
			selected_indexes = []
			for j in range(int(epoch_target)):
				batch_i = random.choice(batch_indexes)
				selected_indexes.append(batch_i)
				batch_indexes.remove(batch_i)
			filenames_index_gen = (index_batch_train[i] for i in selected_indexes)
			
		for i, index_filename in enumerate(filenames_index_gen):

			filenames = [file_list[index] for index in index_filename]
			img_batch_array, label_batch_array = batch_img_gen(filenames, augmented_input_dir, real_input_dir, scale=scale_batch_label)

			loss_value, grads = grad(model, loss, img_batch_array/255, label_batch_array, weight_decay, training=True)

			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			
			epoch_loss_avg.update_state(loss_value)
			
#			epoch_accuracy.update_state(label_batch_array, model(img_batch_array))
			
		# Add eval
		
		if (epoch+1)%10==0:
			model.save('data/model/'+'TinyYolo_%s'%(epoch+1))
			print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format((epoch+1),
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
			if os.path.exists('data/model/'+'TinyYolo_%s'%((epoch+1)-10)):
				shutil.rmtree('data/model/'+'TinyYolo_%s'%((epoch+1)-10))
																
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
