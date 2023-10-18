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
        print('Rotating annotations')
        annot = open(path + image[:-4] + '.txt', 'r')
        print(path + image[:-4] + '.txt')
        annot_lines = annot.readlines()
        print(annot_lines)
        annot.close()
        annot = open(path + image[:-4] +' _90_deg_rotation' + '.txt', 'w')
        for line in annot_lines:
            print(line)
            line = line.split(' ')
            clas = line[0]
            x_center = line[1]
            y_center = line[2]
            width = line[3]
            height = line[4]
            x_center, y_center, width, height = rotate_90_annotation(x_center, y_center, width, height)
            new_line = clas + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n'
            print(new_line)
            annot.write(new_line)
        annot.close()

def rotate_180_degrees(path, image, annot=False):
    img = cv2.imread(path + image)
    img = rotate_180_img(img)
    cv2.imwrite(path + image[:-4] + '_180_deg_rotation.jpg', img)
    if annot == True:
        print('Rotating annotations')
        annot = open(path + image[:-4] + '.txt', 'r')
        print(path + image[:-4] + '.txt')
        annot_lines = annot.readlines()
        print(annot_lines)
        annot.close()
        annot = open(path + image[:-4] +' _180_deg_rotation' + '.txt', 'w')
        for line in annot_lines:
            print(line)
            line = line.split(' ')
            clas = line[0]
            x_center = line[1]
            y_center = line[2]
            width = line[3]
            height = line[4]
            x_center, y_center, width, height = rotate_180_annotation(x_center, y_center, width, height)
            new_line = clas + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n'
            print(new_line)
            annot.write(new_line)
        annot.close()

def rotate_270_degrees(path, image, annot=False):
    img = cv2.imread(path + image)
    img = rotate_270_img(img)
    cv2.imwrite(path + image[:-4] + '_270_deg_rotation.jpg', img)
    if annot == True:
        print('Rotating annotations')
        annot = open(path + image[:-4] + '.txt', 'r')
        print(path + image[:-4] + '.txt')
        annot_lines = annot.readlines()
        print(annot_lines)
        annot.close()
        annot = open(path + image[:-4] +' _270_deg_rotation' + '.txt', 'w')
        for line in annot_lines:
            print(line)
            line = line.split(' ')
            clas = line[0]
            x_center = line[1]
            y_center = line[2]
            width = line[3]
            height = line[4]
            x_center, y_center, width, height = rotate_270_annotation(x_center, y_center, width, height)
            new_line = clas + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n'
            print(new_line)
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

def augment_brightness(image, val, annot=False):
    cv2.imread(image)
    img = increase_brightness(image, val)
    if annot == False:
        cv2.imwrite(image[:-4] + '_brightness_increased.jpg', img)
    else:
        cv2.imwrite(image[:-4] + '_brightness_increased.jpg', img)
        annot = image[:-4] + '.txt'
        annot_new = image[:-4] + '_brightness_increased.txt'
        os.system('cp ' + annot + ' ' + annot_new)

def augment_contrast(image, val, annot=False):
    img = cv2.imread(image)
    img2 = img
    img = increase_contrast(image, val)
    img2 = decrease_contrast(image, val)
    if annot == False:
        cv2.imwrite(image[:-4] + '_contrast_increased.jpg', img)
        cv2.imwrite(image[:-4] + '_contrast_decreased.jpg', img2)
    else:
        cv2.imwrite(image[:-4] + '_contrast_increased.jpg', img)
        cv2.imwrite(image[:-4] + '_contrast_decreased.jpg', img2)
        annot = image[:-4] + '.txt'
        annot_new = image[:-4] + '_contrast_increased.txt'
        annot_new2 = image[:-4] + '_contrast_decreased.txt'
        os.system('cp ' + annot + ' ' + annot_new)
        os.system('cp ' + annot + ' ' + annot_new2)

def augment_saturation(image, val, annot=False):
    img = cv2.imread(image)
    img2 = img
    img = increase_saturation(image, val)
    img2 = decrease_saturation(image, val)
    if annot == False:
        cv2.imwrite(image[:-4] + '_saturation_increased.jpg', img)
        cv2.imwrite(image[:-4] + '_saturation_decreased.jpg', img2)
    else:
        cv2.imwrite(image[:-4] + '_saturation_increased.jpg', img)
        cv2.imwrite(image[:-4] + '_saturation_decreased.jpg', img2)
        annot = image[:-4] + '.txt'
        annot_new = image[:-4] + '_saturation_increased.txt'
        annot_new2 = image[:-4] + '_saturation_decreased.txt'
        os.system('cp ' + annot + ' ' + annot_new)
        os.system('cp ' + annot + ' ' + annot_new2)

def augment_hue(image, val, annot=False):
    img = cv2.imread(image)
    img2 = img
    img = increase_hue(image, val)
    img2 = decrease_hue(image, val)
    if annot == False:
        cv2.imwrite(image[:-4] + '_hue_increased.jpg', img)
        cv2.imwrite(image[:-4] + '_hue_decreased.jpg', img2)
    else:
        cv2.imwrite(image[:-4] + '_hue_increased.jpg', img)
        cv2.imwrite(image[:-4] + '_hue_decreased.jpg', img2)
        annot = image[:-4] + '.txt'
        annot_new = image[:-4] + '_hue_increased.txt'
        annot_new2 = image[:-4] + '_hue_decreased.txt'
        os.system('cp ' + annot + ' ' + annot_new)
        os.system('cp ' + annot + ' ' + annot_new2)

def augment_sharpness(image, val, annot=False):
    img = cv2.imread(image)
    img2 = img
    img = increase_sharpness(image, val)
    img2 = decrease_sharpness(image, val)
    if annot == False:
        cv2.imwrite(image[:-4] + '_sharpness_increased.jpg', img)
        cv2.imwrite(image[:-4] + '_sharpness_decreased.jpg', img2)
    else:
        cv2.imwrite(image[:-4] + '_sharpness_increased.jpg', img)
        cv2.imwrite(image[:-4] + '_sharpness_decreased.jpg', img2)
        annot = image[:-4] + '.txt'
        annot_new = image[:-4] + '_sharpness_increased.txt'
        annot_new2 = image[:-4] + '_sharpness_decreased.txt'
        os.system('cp ' + annot + ' ' + annot_new)
        os.system('cp ' + annot + ' ' + annot_new2)

def augment_blur(image, val, annot=False):
    img = cv2.imread(image)
    img2 = img
    img = increase_blur(image, val)
    img2 = decrease_blur(image, val)
    if annot == False:
        cv2.imwrite(image[:-4] + '_blur_increased.jpg', img)
        cv2.imwrite(image[:-4] + '_decreased_blur.jpg', img2)
    else:
        cv2.imwrite(image[:-4] + '_blur_increased.jpg', img)
        cv2.imwrite(image[:-4] + '_decreased_blur.jpg', img2)
        annot = image[:-4] + '.txt'
        annot_new = image[:-4] + '_blur_increased.txt'
        annot_new2 = image[:-4] + '_decreased_blur.txt'
        os.system('cp ' + annot + ' ' + annot_new)
        os.system('cp ' + annot + ' ' + annot_new2)

def iterate_augment(path, augments, annot=False):
    for image in tqdm.tqdm(os.listdir(path)):
        if image.endswith('.jpg'):
            print(image)
            if 'Rotate 90 degrees' in augments:
                rotate_90_degrees(path, image, annot)
            if 'Rotate 180 degrees' in augments:
                rotate_180_degrees(path, image, annot)
            if 'Rotate 270 degrees' in augments:
                rotate_270_degrees(path, image, annot)
            if 'Play with sharpness' in augments:
                augment_sharpness(image, 10, annot)  
            if 'Play with brightness' in augments:
                augment_brightness(image, 10, annot)
            if 'Play with contrast' in augments:
                augment_contrast(image, 10, annot)
            if 'Play with saturation' in augments:
                augment_saturation(image, 10, annot)
            if 'Play with hue' in augments:
                augment_hue(image, 10, annot)
            if 'Play with blur' in augments:
                augment_blur(image, 10, annot)
