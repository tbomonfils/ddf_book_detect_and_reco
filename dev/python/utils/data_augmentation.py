import cv2
import PIL

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import random


def transform(image,
              translation=(0, 0, 0),
              rotation=(0, 0, 0),
              scaling=(1, 1, 1),
              shearing=(0, 0, 0)):
    
    # get the values on each axis
    t_x, t_y, t_z = translation
    r_x, r_y, r_z = rotation
    sc_x, sc_y, sc_z = scaling
    sh_x, sh_y, sh_z = shearing
    
    # convert degree angles to rad
    theta_rx = np.deg2rad(r_x)
    theta_ry = np.deg2rad(r_y)
    theta_rz = np.deg2rad(r_z)
    theta_shx = np.deg2rad(sh_x)
    theta_shy = np.deg2rad(sh_y)
    theta_shz = np.deg2rad(sh_z)
    
    # get the height and the width of the image
    h, w = image.shape[:2]
    # compute its diagonal
    diag = (h ** 2 + w ** 2) ** 0.5
    # compute the focal length
    f = diag
    if np.sin(theta_rz) != 0:
        f /= 2 * np.sin(theta_rz)
        
    # set the image from cartesian to projective dimension
    H_M = np.array([[1, 0, -w / 2],
                    [0, 1, -h / 2],
                    [0, 0, 1],
                    [0, 0, 1]])

    """
            We will define our matrices here in next parts
                                                            """
    Identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    
    # adjust the translation on z
    t_z = (f - t_z) / sc_z ** 2
    # translation matrix to translate the image
    T_M = np.array([[1, 0, 0, t_x],
                    [0, 1, 0, t_y],
                    [0, 0, 1, t_z],
                    [0, 0, 0,  1]])
    
    # calculate cos and sin of angles
    sin_rx, cos_rx = np.sin(theta_rx), np.cos(theta_rx)
    sin_ry, cos_ry = np.sin(theta_ry), np.cos(theta_ry)
    sin_rz, cos_rz = np.sin(theta_rz), np.cos(theta_rz)
    
    # get the rotation matrix on x axis
    R_Mx = np.array([[1,      0,       0, 0],
                     [0, cos_rx, -sin_rx, 0],
                     [0, sin_rx,  cos_rx, 0],
                     [0,      0,       0, 1]])
    # get the rotation matrix on y axis
    R_My = np.array([[cos_ry, 0, -sin_ry, 0],
                     [     0, 1,       0, 0],
                     [sin_ry, 0,  cos_ry, 0],
                     [     0, 0,       0, 1]])
    # get the rotation matrix on z axis
    R_Mz = np.array([[cos_rz, -sin_rz, 0, 0],
                     [sin_rz,  cos_rz, 0, 0],
                     [     0,       0, 1, 0],
                     [     0,       0, 0, 1]])
    # compute the full rotation matrix
    R_M = np.dot(np.dot(R_Mx, R_My), R_Mz)
    
    # get the scaling matrix
    Sc_M = np.array([[sc_x,     0,    0, 0],
                     [   0,  sc_y,    0, 0],
                     [   0,     0, sc_z, 0],
                     [   0,     0,    0, 1]])
    
    # get the tan of angles
    tan_shx = np.tan(theta_shx)
    tan_shy = np.tan(theta_shy)
    tan_shz = np.tan(theta_shz)
    # get the shearing matrix on x axis
    Sh_Mx = np.array([[      1, 0, 0, 0],
                      [tan_shy, 1, 0, 0],
                      [tan_shz, 0, 1, 0],
                      [      0, 0, 0, 1]])
    # get the shearing matrix on y axis
    Sh_My = np.array([[1, tan_shx, 0, 0],
                      [0,       1, 0, 0],
                      [0, tan_shz, 1, 0],
                      [0,       0, 0, 1]])
    # get the shearing matrix on z axis
    Sh_Mz = np.array([[1, 0, tan_shx, 0],
                      [0, 1, tan_shy, 0],
                      [0, 0,       1, 0],
                      [0, 0,       0, 1]])
    # compute the full shearing matrix
    Sh_M = np.dot(np.dot(Sh_Mx, Sh_My), Sh_Mz)
    
    # set the image projective to carrtesian dimension
    Hp_M = np.array([[f, 0, w/2 , 0],
                     [0, f, h/2 , 0],
                     [0, 0, 1, 0]])
    
    # compute the full transform matrix
    M = Identity
    M = np.dot(Sh_M, M)
    M = np.dot(Sc_M, M)
    M = np.dot(R_M,  M)
    M = np.dot(T_M,  M)
    M = np.dot(Hp_M, np.dot(M, H_M))
    # apply the transformation
    return M
	
def get_image_corners(img, return_center=True):

	img_corners = [[0, 0],
	[img.shape[1], 0],
	[0, img.shape[0]],
	img.shape[:2][::-1]]
	
	if return_center:
		img_corners.append(np.array(img.shape[:2][::-1])/2)
	return np.array(img_corners)
	
def project_point(point, M):
	x, y = point
	projected_point = (int(round((M[0][0]*x + M[0][1]*y + M[0][2])/(M[2][0]*x + M[2][1]*y + M[2][2]))),
	int(round((M[1][0]*x + M[1][1]*y + M[1][2])/(M[2][0]*x + M[2][1]*y + M[2][2]))))
	return projected_point
	
def find_translation(points):
	translation = np.abs(np.multiply(np.min(points, axis=0), (np.min(points, axis=0)<0).T))
	return translation
	
def find_output_size(points, translation):
	output_size = np.max(points, axis=0)
	return output_size

def create_img_mask(img):
	img_mask = ((np.sum(img, axis=2)!=0)*255).astype(np.uint8)
	return img_mask
	
def draw_circles(img, points, radius=10, color=(255,0,0), thickness=2):
	for point in points:
		center_coordinates = tuple(point)
		img = cv2.circle(img, center_coordinates, radius, color, thickness)
	return img
	
def get_target_width_height(cover_img, background_img):
    ratio = random.uniform(0.5, 3/4)
    target_height = int(np.round(background_img.size[1]*ratio))
    width_ratio = target_height/cover_img.shape[0]
    target_width = int(width_ratio*cover_img.shape[1])
    return target_width, target_height

def scale_cover_image(cover_img, background_img):
    img_live_dim = get_target_width_height(cover_img, background_img)
    return np.array(Image.fromarray(cover_img.copy()).resize(img_live_dim))
	
def define_position_of_cover_in_background(cover_img_projected_center_coord, background_img, offset_ratio=0.1):
	center_background_img = (np.array(background_img.size)/2).astype(np.int)
	
	offset = np.round(np.array([random.uniform(-offset_ratio*center_background_img[0],
	offset_ratio*center_background_img[0]),
	random.uniform(-offset_ratio*center_background_img[1],
	offset_ratio*center_background_img[1])])).astype(np.int)
	
	position = tuple(center_background_img - cover_img_projected_center_coord + offset)
	return position
	
def post_composition_augmentation(image_array):
	aug = iaa.Cutout(fill_mode="constant", cval=(0, 255),
	fill_per_channel=0.5, nb_iterations=(1, 5), size=0.25, squared=False)
	
	image_array = aug(image=image_array)
	
	enhancer = ImageEnhance.Brightness(Image.fromarray(image_array))
	
	return enhancer.enhance(random.uniform(0.3, 1))
	
def generate_file_path_name(filename_start, projected_points_coord, i,extension='.jpg'):
	filename = '_'.join([filename_start, str(i)]+[str(point) for point in projected_points_coord.reshape((8))])
	return filename+extension
	
def read_augmented_img_filename(file_path):
	img = PIL.Image.open(file_path)
	points = np.array(file_path.split('/')[-1].split('.')[0].split('_')[-8:]).reshape((4, 2)).astype(np.int)
	return img, points
	
def end_to_end_transformation(cover_img, background_img, output_size=(512, 512)):
	background_img = background_img.resize(output_size)
	rotation = (random.uniform(0, 45), random.uniform(0, 45), random.uniform(0, 45))
	
	resized_cover_img = scale_cover_image(cover_img, background_img)
	cover_img_corners = get_image_corners(resized_cover_img)
	
	M = transform(resized_cover_img,
			  translation=(0, 0, 0),
			  rotation=rotation,
			  scaling=(1, 1, 1),
			  shearing=(0, 0, 0))
	proj_cover_img_corners = np.array([project_point(point, M) for point in cover_img_corners])
	
	translation = find_translation(proj_cover_img_corners)
	
	j=0
	while (proj_cover_img_corners<0).any():
		translation += j*10
		M = transform(resized_cover_img,
					  translation=(translation[0], translation[1], 0),
					  rotation=rotation,
					  scaling=(1, 1, 1),
					  shearing=(0, 0, 0))
		proj_cover_img_corners = np.array([project_point(point, M) for point in cover_img_corners])
		j+=1
	
	warped_cover_img_size = find_output_size(proj_cover_img_corners, translation)
	warped_cover_img = cv2.warpPerspective(resized_cover_img, M, tuple(warped_cover_img_size))
	
	warped_cover_img_mask = create_img_mask(warped_cover_img)
	
	center_background_img = (np.array(background_img.size)/2).astype(np.int)
	
	position_cover_img_in_background = define_position_of_cover_in_background(proj_cover_img_corners[-1],
	background_img)
	
	background_img.paste(Image.fromarray(warped_cover_img[:, :, ::-1]),
	position_cover_img_in_background,
	Image.fromarray(warped_cover_img_mask))
	augmented_img = background_img.copy()
	
	augmented_img_array = post_composition_augmentation(np.array(augmented_img))
	
	return augmented_img_array, proj_cover_img_corners[:-1]+position_cover_img_in_background