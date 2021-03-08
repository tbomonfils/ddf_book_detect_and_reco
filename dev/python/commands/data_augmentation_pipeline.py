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
	parser.add_argument('--cover_input_path', type=str, default='data/input/book_covers/',
						help='The path to the directory where the cover images to be augmented are')
	parser.add_argument('--background_input_path', type=str, default='data/input/shops_atmosphere/',
                        help='The path to the directory where the background images to be used are')
	parser.add_argument('--augmented_image_dir', type=str, default='data/output/tests/augmented_covers/',
					help='The path to the data files to be augmented')
	return parser.parse_args(args)
	
def pipeline(cover_input_path='data/input/book_covers/',
	background_input_path='data/input/shops_atmosphere/',
	augmented_image_dir='data/output/tests/augmented_covers/'):
	
	cover_img_dir = cover_input_path
	background_img_dir = background_input_path
	
	#Hack to complete generationof image
	cover_img_cat_list = os.listdir(cover_img_dir)
	for e in ['main_dataset.csv']:
		cover_img_cat_list.remove(e)
	
	background_img_nb = len(os.listdir(background_img_dir))-1
	
	for cover_img_cat in cover_img_cat_list:
		print(cover_img_cat)
		for cover_img_file in os.listdir(cover_img_dir+cover_img_cat):
			cover_img_path = cover_img_dir+cover_img_cat+'/'+cover_img_file
			cover_img = cv2.imread(cover_img_path)
			
			for i in range(10):
				
				background_img_file = os.listdir(background_img_dir)[random.randint(0, background_img_nb)]
				background_img_path = background_img_dir+background_img_file
				background_img = PIL.Image.open(background_img_path).resize((512, 512))
				
				rotation = (random.uniform(0, 45), random.uniform(0, 45), random.uniform(0, 45))
					
				try:
					
					resized_cover_img = da.scale_cover_image(cover_img, background_img)
					cover_img_corners = da.get_image_corners(resized_cover_img)
					
					M = da.transform(resized_cover_img,
							  translation=(0, 0, 0),
							  rotation=rotation,
							  scaling=(1, 1, 1),
							  shearing=(0, 0, 0))
					proj_cover_img_corners = np.array([da.project_point(point, M) for point in cover_img_corners])
					
					translation = da.find_translation(proj_cover_img_corners)
					
					j=0
					while (proj_cover_img_corners<0).any():
						translation += j*10
						M = da.transform(resized_cover_img,
									  translation=(translation[0], translation[1], 0),
									  rotation=rotation,
									  scaling=(1, 1, 1),
									  shearing=(0, 0, 0))
						proj_cover_img_corners = np.array([da.project_point(point, M) for point in cover_img_corners])
						j+=1
					
					warped_cover_img_size = da.find_output_size(proj_cover_img_corners, translation)
					warped_cover_img = cv2.warpPerspective(resized_cover_img, M, tuple(warped_cover_img_size))
					
					warped_cover_img_mask = da.create_img_mask(warped_cover_img)
					
					center_background_img = (np.array(background_img.size)/2).astype(np.int)
					
					position_cover_img_in_background = da.define_position_of_cover_in_background(proj_cover_img_corners[-1],
					background_img)
					
					background_img.paste(Image.fromarray(warped_cover_img[:, :, ::-1]),
					position_cover_img_in_background,
					Image.fromarray(warped_cover_img_mask))
					augmented_img = background_img.copy()
					
					augmented_img_array = da.post_composition_augmentation(np.array(augmented_img))
						
					augmented_img_output_dir = augmented_image_dir
					filename = da.generate_file_path_name(cover_img_cat+'_'+cover_img_file.split('.')[0], proj_cover_img_corners[:-1]+position_cover_img_in_background, i)
					augmented_img_output_path = augmented_img_output_dir+filename
					
					augmented_img_array.convert('RGB').save(augmented_img_output_path)
				except Exception as e:
					with(open('logs/logs.txt', 'a'))as f:
						f.write(cover_img_path+', '+background_img_path+', '+'_'.join([str(angle)for angle in rotation])+'\n')
					
	
class Command():

	def run(self, args):
		parsed_args = arg_parser(args)
		
		cover_input_path = parsed_args.cover_input_path
		background_input_path = parsed_args.background_input_path
		augmented_image_dir = parsed_args.augmented_image_dir
		
		pipeline(cover_input_path, background_input_path, augmented_image_dir)