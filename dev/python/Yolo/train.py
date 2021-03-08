import os
import argparse

import numpy as np
import tensorflow as tf
import random

import PIL.Image as Image

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for using ScaledYOLOv4.')
    #save model
    parser.add_argument('--output-model-dir', default='./output_model')
    #training
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--start-eval-epoch', default=10, type=int)
    parser.add_argument('--eval-epoch-interval', default=1)
    #model
    parser.add_argument('--model-type', default='tiny', help="choices=['tiny','p5','p6','p7']")
    parser.add_argument('--use-pretrain', default=False, type=bool)
    parser.add_argument('--tiny-coco-pretrained-weights',
                        default='./pretrain/ScaledYOLOV4_tiny_coco_pretrain/coco_pretrain')
    parser.add_argument('--p5-coco-pretrained-weights',
                        default='./pretrain/ScaledYOLOV4_p5_coco_pretrain/coco_pretrain')
    parser.add_argument('--p6-coco-pretrained-weights',
                        default='./pretrain/ScaledYOLOV4_p6_coco_pretrain/coco_pretrain')
    parser.add_argument('--checkpoints-dir', default='./checkpoints',help="Directory to store  checkpoints of model during training.")
    #loss
    parser.add_argument('--box-regression-loss', default='ciou',help="choices=['giou','diou','ciou']")
    parser.add_argument('--classification-loss', default='bce', help="choices=['ce','bce','focal']")
    parser.add_argument('--focal-alpha', default= 0.25)
    parser.add_argument('--focal-gamma', default=2.0)
    parser.add_argument('--ignore-thr', default=0.7)
    parser.add_argument('--reg-losss-weight', default=0.05)
    parser.add_argument('--obj-losss-weight', default=1.0)
    parser.add_argument('--cls-losss-weight', default=0.5)
    #dataset
    parser.add_argument('--dataset-type', default='voc', help="voc,coco")
    parser.add_argument('--num-classes', default=20)
    parser.add_argument('--class-names', default='voc.names', help="voc.names,coco.names")
    parser.add_argument('--dataset', default='/home/wangem1/dataset/VOC2007_2012')#
    parser.add_argument('--voc-train-set', default='VOC2007,trainval,VOC2012,trainval')
    parser.add_argument('--voc-val-set', default='VOC2007,test')
    parser.add_argument('--voc-skip-difficult', default=True)
    parser.add_argument('--coco-train-set', default='train2017')
    parser.add_argument('--coco-valid-set', default='val2017')
    '''
    voc dataset directory:
        VOC2007
                Annotations
                ImageSets
                JPEGImages
        VOC2012
                Annotations
                ImageSets
                JPEGImages
    coco dataset directory:
        annotations/instances_train2017.json
        annotations/instances_val2017.json
        images/train2017
        images/val2017
    '''
    parser.add_argument('--augment', default='mosaic',help="choices=[None,'only_flip_left_right','ssd_random_crop','mosaic']")
    parser.add_argument('--multi-scale', default='416',help="Input data shapes for training, use 320+32*i(i>=0)")#896
    parser.add_argument('--max-box-num-per-image', default=100)
    #optimizer
    parser.add_argument('--optimizer', default='sgd', help="choices=[adam,sgd]")
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--nesterov', default=True)
    parser.add_argument('--weight-decay', default=5e-4)
    #lr scheduler
    parser.add_argument('--lr-scheduler', default='warmup_cosinedecay', type=str, help="choices=['step','warmup_cosinedecay']")
    parser.add_argument('--init-lr', default=1e-3, type=float)
    parser.add_argument('--lr-decay', default=0.1, type=float)
    parser.add_argument('--lr-decay-epoch', default=[160, 180])
    parser.add_argument('--warmup-epochs', default=10, type=int)
    parser.add_argument('--warmup-lr', default=1e-6, type=float)
    #postprocess
    parser.add_argument('--nms', default='diou_nms', help="choices=['hard_nms','diou_nms']")
    parser.add_argument('--nms-max-box-num', default=300)
    parser.add_argument('--nms-iou-threshold', default=0.2, type=float)
    parser.add_argument('--nms-score-threshold', default=0.01, type=float)
    #anchor
    parser.add_argument('--anchor-match-type', default='wh_ratio',help="choices=['iou','wh_ratio']")
    parser.add_argument('--anchor-match-iou_thr', default=0.2, type=float)
    parser.add_argument('--anchor-match-wh-ratio-thr', default=4.0, type=float)

    parser.add_argument('--label-smooth', default=0.0, type=float)
    parser.add_argument('--scales-x-y', default=[2., 2., 2., 2., 2.])
    parser.add_argument('--accumulated-gradient-num', default=1, type=int)

    parser.add_argument('--tensorboard', default=True, type=bool)

    return parser.parse_args(args)
	
def loss(model, x, y, training):
    l = tf.keras.losses.BinaryCrossentropy()
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)

    return l(y_true=y, y_pred=y_)
	
def grad(model, inputs, targets, training):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=training)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)
	
def split_train_test_index(index_size, breakdown_ratio=0.8):
	index_split = int((index_size*breakdown_ratio//64+1)*64)

	index_array = np.array([i for i in range(index_size)])
	random.shuffle(index_array)
	
	index_train, index_test = index_array[:index_split], index_array[index_split:]
	
	return index_train, index_test
	
#def gen_img_batch_generator(file_dir_input):
#	file_list = os.listdir(file_dir)
#	nb_obs = len(file_list)
#	
#	index_batch_train, index_batch_test = split_train_test_index(nb_obs)
#	img_batch_index = index_batch_train.shape[0]
#	
#	img_batch = []
#	label_batch = []
#			
#	for index in img_batch_train:
#		for i in index:
#			filename = file_list[i]
#			img_batch.append(np.array(Image.open(file_dir+filename)))
#			
#			points = np.array(filename.split('.')[0].split('_')[-8:]).reshape((4, 2)).astype(np.int)
#			label_batch.append(np.concatenate((np.concatenate(np.array(list(zip(points[:,0]/img_array.shape[0], \
#			points[:,1]/img_array.shape[1])))), np.array([1]))))
#	
#	return (file_list[] for index in img_batch_train)

def train_model(model, loss, nb_epoch=100, batch_size=64, input_dir='data/output/tests/augmented_covers/', output_dir='data/model/', breakdown_ratio=0.8):
	
	optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
	
	file_list = os.listdir(input_dir)
	nb_obs = len(file_list)
	
	index_train, index_test = split_train_test_index(nb_obs)

	test_index_split = index_test.shape[0]//batch_size
	index_batch_test = index_test[:test_index_split*batch_size].reshape((test_index_split, batch_size))
	
	for epoch in range(nb_epoch):
		epoch_loss_avg = tf.keras.metrics.Mean()
		epoch_accuracy = tf.keras.metrics.BinaryCrossentropy()


		random.shuffle(index_train)
		index_batch_train = index_train.reshape((index_train.shape[0]//batch_size, batch_size))
		
		for i, index_batch in enumerate(index_batch_train):
			print(epoch, i)
			img_batch = []
			label_batch = []
			for index in index_batch:
				filename = file_list[index]
				img_batch.append(np.array(Image.open(input_dir+filename)))

				points = np.array(filename.split('.')[0].split('_')[-8:]).reshape((4, 2)).astype(np.int)
				label_batch.append(np.concatenate((np.concatenate(np.array(list(zip(points[:,0]/img_batch[-1].shape[-3], points[:,1]/img_batch[-1].shape[-2])))), np.array([1]))))

			img_batch_array = np.stack(img_batch)
			label_batch_array = np.stack(label_batch)

			loss_value, grads = grad(model, img_batch_array, label_batch_array, training=True)

			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			
			epoch_loss_avg.update_state(loss_value)
			
			
			epoch_accuracy.update_state(label_batch_array, model(img_batch_array))
			
		# Add eval
		
		if (epoch+1)%10==0:
			model.save('data/model/'+'TinyYolo_%s'%epoch)
			print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
