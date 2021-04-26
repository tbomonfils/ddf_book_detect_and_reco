import argparse
import os
import sys

import cv2
from dev.python.utils import data_augmentation as da
import numpy as np
import PIL
import PIL.Image as Image
import random


def arg_parser(args):
	parser = argparse.ArgumentParser('Parsing data augmentation pipeline arguments')
	parser.add_argument('--cover_input_dir', type=str, default='data/input/book_covers/',
						help='The path to the directory where the cover images to be augmented are')
	parser.add_argument('--background_input_dir', nargs='*',
                        help='The path to the directory where the background images to be used are')
	parser.add_argument('--augmented_image_train_dir', type=str, default='data/output/tests/augmented_covers_train/',
					help='The path to where the augmented data will be saved for training')
	parser.add_argument('--augmented_image_test_dir', type=str, default='data/output/tests/augmented_covers_test/',
					help='The path to where the augmented data will be saved for testing')
	return parser.parse_args(args)


def rdm_dir_selection(input_dirs, file_distrib):
	rand_float = random.uniform(0, 1)
	j=0
	while file_distrib[j]<rand_float:
		j+=1
	return j, input_dirs[j]

	
def pipeline(cover_input_dir='data/input/book_covers/',
	background_input_dir='data/input/shops_atmosphere',
	augmented_image_train_dir='data/output/tests/augmented_covers_train/',
	augmented_image_test_dir='data/output/tests/augmented_covers_test/',
	threshold_split=0.8):
	
	#Hack to complete generationof image
	cover_img_cat_list = os.listdir(cover_input_dir)
	for e in ['main_dataset.csv']:
		cover_img_cat_list.remove(e)
	
	background_nb_file_per_dir = [len(os.listdir(directory)) for directory in background_input_dir]

	for cover_img_cat in cover_img_cat_list:
		print(cover_img_cat)
		for cover_img_file in os.listdir(cover_input_dir+cover_img_cat):
			
			if '.csv' in cover_img_file:
				continue
			else:
				
				cover_img_path = cover_input_dir+cover_img_cat+'/'+cover_img_file
				cover_img = cv2.imread(cover_img_path)
				
				for i in range(10):
					
					j, background_img_dir = rdm_dir_selection(background_input_dir,
											[sum(background_nb_file_per_dir[:i+1])/sum(background_nb_file_per_dir)for i, n in enumerate(background_nb_file_per_dir)])
					background_img_file = os.listdir(background_img_dir)[random.randint(0, background_nb_file_per_dir[j]-1)]
					background_img_path = background_img_dir+background_img_file
					background_img = PIL.Image.open(background_img_path)
					
					try:
						augmented_img, points = da.end_to_end_transformation(cover_img, background_img)
					except Exception as e:
						with(open('logs/logs.txt', 'a'))as f:
							f.write(cover_img_path+', '+background_img_path+'\n')
							
					_, augmented_img_output_dir = rdm_dir_selection([augmented_image_train_dir, augmented_image_test_dir], file_distrib=[threshold_split, 1])
					filename = da.generate_file_path_name(cover_img_cat+'_'+cover_img_file.split('.')[0], points, i)
					
					augmented_img_output_path = augmented_img_output_dir+filename
					
					augmented_img.convert('RGB').save(augmented_img_output_path)
					
	
class Command():

	def run(self, args):
		parsed_args = arg_parser(args)
		
		cover_input_dir = parsed_args.cover_input_dir
		background_input_dir = parsed_args.background_input_dir
		augmented_image_train_dir = parsed_args.augmented_image_train_dir
		augmented_image_test_dir = parsed_args.augmented_image_test_dir
		
		pipeline(cover_input_dir, background_input_dir, augmented_image_train_dir, augmented_image_test_dir)