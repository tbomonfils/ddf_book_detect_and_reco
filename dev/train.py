import sys
import argparse

import tensorflow as tf

from dev.python.Yolo import model_architecture as m_a
from dev.python.Yolo import train


def parse_args(args):
	parser = argparse.ArgumentParser(description='Simple training script for using ScaledYOLOv4.')
	#save model
	parser.add_argument('--output-model-dir', default='./output_model')
	#training
	parser.add_argument('--epochs', default=300, type=int)
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--start-eval-epoch', default=10, type=int)
	parser.add_argument('--eval-epoch-interval', default=1)
	parser.add_argument('--epoch_target', default='full')

	return parser.parse_args(args)
	
def main(args):
	model = m_a.Yolov4_tiny(args, training=True)
	train.train_model(model, tf.keras.losses.BinaryCrossentropy(), nb_epoch=args.epochs, batch_size=args.batch_size)
	
if __name__ == "__main__":
	args = parse_args(sys.argv[1:])
	main(args)