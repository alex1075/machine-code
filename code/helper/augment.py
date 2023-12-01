import os
import sys
import cv2
import tqdm
import inquirer
from code.helper.annotations import *
from code.helper.utils import *
from code.helper.imageTools import *


def choose_augmentations():
    question = [inquirer.Checkbox('augmentations',
                            message="Do you want to use augmentations on the dataset?",
                            choices=['Rotate 90 degrees', 'Rotate 180 degrees', 'Rotate 270 degrees',  
                                     'Play with sharpness', 'Play with contrast', 'Play with brightness',
                                     'Play with saturation', 'Play with hue', 'Play with blur',],
                        ),]
    answer = inquirer.prompt(question)
    return answer['augmentations']

def rotate_90_degrees(path, image, annot=False):
    img = cv2.imread(path + image)
    img = rotate_90_img(img)
    cv2.imwrite(path + image[:-4] + '_90_deg_rotation.jpg', img)
    if annot == True:
        annot = open(path + image[:-4] + '.txt', 'r')
        annot_lines = annot.readlines()
        annot.close()
        annot = open(path + image[:-4] +'_90_deg_rotation' + '.txt', 'w')
        for line in annot_lines:
            line = line.strip('\n')
            line = line.split(' ')
            clas = line[0].strip('\n')
            x_center = line[1].strip('\n')
            y_center = line[2].strip('\n')
            width = line[3].strip('\n')
            height = line[4].strip('\n')
            x_center, y_center, width, height = rotate_90_annotation(x_center, y_center, width, height)
            # new_line = clas + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n'
            new_line = f'{clas} {x_center} {y_center} {width} {height} \n'
            annot.write(new_line)
        annot.close()

def rotate_180_degrees(path, image, annot=False):
    img = cv2.imread(path + image)
    img = rotate_180_img(img)
    cv2.imwrite(path + image[:-4] + '_180_deg_rotation.jpg', img)
    if annot == True:
        annot = open(path + image[:-4] + '.txt', 'r')
        annot_lines = annot.readlines()
        annot.close()
        annot = open(path + image[:-4] +'_180_deg_rotation' + '.txt', 'w')
        for line in annot_lines:
            line = line.strip('\n')
            line = line.split(' ')
            clas = line[0].strip('\n')
            x_center = line[1].strip('\n')
            y_center = line[2].strip('\n')
            width = line[3].strip('\n')
            height = line[4].strip('\n')
            x_center, y_center, width, height = rotate_180_annotation(x_center, y_center, width, height)
            # new_line = clas + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n'
            new_line = f'{clas} {x_center} {y_center} {width} {height} \n'
            annot.write(new_line)
        annot.close()

def rotate_270_degrees(path, image, annot=False):
    img = cv2.imread(path + image)
    img = rotate_270_img(img)
    cv2.imwrite(path + image[:-4] + '_270_deg_rotation.jpg', img)
    if annot == True:
        annot = open(path + image[:-4] + '.txt', 'r')
        annot_lines = annot.readlines()
        annot.close()
        annot = open(path + image[:-4] +'_270_deg_rotation' + '.txt', 'w')
        for line in annot_lines:
            line = line.strip('\n')
            line = line.split(' ')
            clas = line[0].strip('\n')
            x_center = line[1].strip('\n')
            y_center = line[2].strip('\n')
            width = line[3].strip('\n')
            height = line[4].strip('\n')
            x_center, y_center, width, height = rotate_270_annotation(x_center, y_center, width, height)
            new_line = f'{clas} {x_center} {y_center} {width} {height} \n'
            annot.write(new_line)
        annot.close()

def rotate_90_img(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def rotate_180_img(img):
    return cv2.rotate(img, cv2.ROTATE_180)

def rotate_270_img(img):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def rotate_90_annotation(x1, y1, x2, y2):
    bbox = [x1, y1, x2, y2]
    bbox = rotate_boxes90(bbox)
    return bbox[0], bbox[1], bbox[2], bbox[3]
def rotate_180_annotation(x1, y1, x2, y2):
    bbox = [x1, y1, x2, y2]
    bbox = rotate_boxes180(bbox)
    return bbox[0], bbox[1], bbox[2], bbox[3]

def rotate_270_annotation(x1, y1, x2, y2):
    bbox = [x1, y1, x2, y2]
    bbox = rotate_boxes270(bbox)
    return bbox[0], bbox[1], bbox[2], bbox[3]

def augment_brightness(path, image, beta=5, annot=False):
    img =cv2.imread(path + image)
    img = contrast_n_brightness(img, 1.0, 0, 0, beta)
    if annot == False:
        cv2.imwrite(path + image[:-4] + '_brightness_increased.jpg', img)
    else:
        cv2.imwrite(path + image[:-4] + '_brightness_increased.jpg', img)
        annot = path + image[:-4] + '.txt'
        annot_new = path + image[:-4] + '_brightness_increased.txt'
        os.system('cp ' + annot + ' ' + annot_new)

def augment_contrast(path, image, alpha=1.5, annot=False):
    img = cv2.imread(path + image)
    img2 = img
    img = contrast_n_brightness(img, alpha, 0, 0, 0)
    if annot == False:
        cv2.imwrite(path + image[:-4] + '_contrast_increased.jpg', img)
    else:
        cv2.imwrite(path + image[:-4] + '_contrast_increased.jpg', img)
        annot = path + image[:-4] + '.txt'
        annot_new = path + image[:-4] + '_contrast_increased.txt'
        os.system('cp ' + annot + ' ' + annot_new)

def augment_saturation(path, image, val, annot=False):
    img = cv2.imread(path + image)
    img = increase_saturation(img, val)
    if annot == False:
        cv2.imwrite(path + image[:-4] + '_saturation_increased.jpg', img)
    else:
        cv2.imwrite(path + image[:-4] + '_saturation_increased.jpg', img)
        annot = path + image[:-4] + '.txt'
        annot_new = path + image[:-4] + '_saturation_increased.txt'
        os.system('cp ' + annot + ' ' + annot_new)

def augment_hue(path, image, val, annot=False):
    img = cv2.imread(path + image)
    img = increase_hue(img, val)
    if annot == False:
        cv2.imwrite(path + image[:-4] + '_hue_increased.jpg', img)
    else:
        cv2.imwrite(path + image[:-4] + '_hue_increased.jpg', img)
        annot = path + image[:-4] + '.txt'
        annot_new = path + image[:-4] + '_hue_increased.txt'
        os.system('cp ' + annot + ' ' + annot_new)

def augment_sharpness(path, image, annot=False):
    img = cv2.imread(path + image)
    img = increase_sharpness(img)
    if annot == False:
        cv2.imwrite(path + image[:-4] + '_sharpness_increased.jpg', img)
    else:
        cv2.imwrite(path + image[:-4] + '_sharpness_increased.jpg', img)
        annot = path + image[:-4] + '.txt'
        annot_new = path + image[:-4] + '_sharpness_increased.txt'
        os.system('cp ' + annot + ' ' + annot_new)

def augment_blur(path, image, val, annot=False):
    img = cv2.imread(path +image)
    img2 = img
    img = increase_gaussian_blur(img, val)
    img2 = increase_median_blur(img2, val)
    if annot == False:
        cv2.imwrite(path + image[:-4] + '_gaussian_blur.jpg', img)
        cv2.imwrite(path + image[:-4] + '_median_blur.jpg', img2)
    else:
        cv2.imwrite(path + image[:-4] + '_gaussian_blur.jpg', img)
        cv2.imwrite(path + image[:-4] + '_median_blur.jpg', img2)
        annot = path + image[:-4] + '.txt'
        annot_new = path + image[:-4] + '_gaussian_blur.txt'
        annot_new2 = path + image[:-4] + '_median_blur.txt'
        os.system('cp ' + annot + ' ' + annot_new)
        os.system('cp ' + annot + ' ' + annot_new2)

def iterate_augment(path, augments, annot=False):
    for image in tqdm.tqdm(os.listdir(path)):
        if image.endswith('.jpg'):
            if 'Rotate 90 degrees' in augments:
                rotate_90_degrees(path, image, annot) #works
            if 'Rotate 180 degrees' in augments:
                rotate_180_degrees(path, image, annot) #works
            if 'Rotate 270 degrees' in augments:
                rotate_270_degrees(path, image, annot) #works
            if 'Play with sharpness' in augments:
                augment_sharpness(path, image, annot)  #works
            if 'Play with brightness' in augments:
                augment_brightness(path, image, 10, annot) #works
            if 'Play with contrast' in augments:
                augment_contrast(path, image, 1.5, annot) #works
            if 'Play with saturation' in augments:
                augment_saturation(path, image, 1.5, annot) #works
            if 'Play with hue' in augments:
                augment_hue(path, image, 0.7, annot) # works
            if 'Play with blur' in augments:
                augment_blur(path, image, 5, annot) #works