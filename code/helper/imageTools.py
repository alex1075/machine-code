import os
import cv2
import tqdm
import fnmatch
import numpy as np
import shutil
from code.helper.utils import *
from code.helper.annotations import *

# Sharpen image using an unsharp mask
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def imgSizeCheck(image, path, x, y):
    img = cv2.imread(path + image)
    height, width, channels = img.shape
    if height << y:
        diff = y - height
        difftoo = x - width
        corrected_img = cv2.copyMakeBorder(img, 0, diff, 0, difftoo,  cv2.BORDER_CONSTANT, value=[0,0,0])
        cv2.imwrite(path + image[:-4] + ".jpg", corrected_img)
    elif width << x:
        diff = y - height
        difftoo = x - width
        corrected_img = cv2.copyMakeBorder(img, 0, diff, 0, difftoo,  cv2.BORDER_CONSTANT, value=[0,0,0])
        cv2.imshow(corrected_img)
        cv2.imwrite(path + image[:-4] + ".jpg", corrected_img)
    else:
        pass

# crop images in chunks of size (x,y) and adapt annotations
def crop_images(x, y, path, save_path, annotations=True):
    if annotations == True:
        shutil.copy(path + "classes.txt", save_path)
    else:
        pass
    images = os.listdir(path)
    for image in tqdm.tqdm(images, desc="Cropping images"):
        if image.endswith(".jpg"):
            # print('Tick')
            img = cv2.imread(path + image)
            height, width, channels = img.shape
            for i in range(0, height, y):
                for j in range(0, width, x):
                    crop_img = img[i:i+y, j:j+x]
                    new_name = image[:-4] + '_' + str(i) + '_' + str(j)
                    cv2.imwrite(save_path + new_name + ".jpg", crop_img)
                    if annotations == True:
                        change_annotation(i, j, x, y, height, width, path, image, new_name, save_path)
                    else:
                        pass
                img = cv2.imread(path + image)
            height, width, channels = img.shape
            for i in range(0, height, y):
                 for j in range(0, width, x):
                    crop = img[i:i+y, j:j+x]
                    cv2.imwrite(save_path + image[:-4] + '_' + str(i) + '_' + str(j) + '.jpg', crop)
        else:
            # print('Tock')
            pass

def checkAllImg(path, x, y):
    images = os.listdir(path)
    for image in tqdm.tqdm(images):
        if image.endswith(".jpg"):
            imgSizeCheck(image, path, x, y)

def del_top_n_bottom_parts(path):
        for file in os.listdir(path):
            if fnmatch.fnmatch(file, '*_0_*') or fnmatch.fnmatch(file, '*_1664_*'):
                os.remove(path + file)



def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def background_removal_with_alpha_sub(original_img, bacground_img, alpha=0.5):
    img = cv2.imread(original_img)
    background = cv2.imread(bacground_img)
    img = img.astype('f')
    background = background.astype('f')
    out = (alpha * ( img -  background ) + 128).clip(0, 255)
    out = np.around(out, decimals=0)
    out = out.astype(np.uint8)
    return out

def background_removal_with_alpha_div(original_img, bacground_img, alpha=0.5):
    img = cv2.imread(original_img)
    background = cv2.imread(bacground_img)
    img = img.astype('f')
    background = background.astype('f')
    out =  ((( img /  background ) * 125)).clip(0, 255)
    out = np.around(out, decimals=0)
    out = out.astype(np.uint8)
    return out

def background_removal_with_alpha(original_img, bacground_img, alpha=0.5):
    out = background_removal_with_alpha_sub(original_img, bacground_img, alpha)
    return out


def blur_bbox_etended(file, annotations, save_path):
    """Blurs the bounding box, and pixels around it, of the image.
    
    Args:
        img: The image to blur.
        annotations: bounding box coordinates
        save_path: path to save the image
        
    Returns:
        The image with the bounding boxes blurred.
    """
    bbox = []
    temp = file.split('/')
    name = temp[-1]
    img = cv2.imread(file)
    for i in annotations:
        ll = float(i) * 416
        bbox. append(int(ll))
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    tx = x - (w/2) - 10
    ty = y + (h/2) + 10
    bx = x + (w/2) + 10
    by = y - (h/2) - 10
    blur_x = int(tx)
    blur_y = int(ty)
    blur_width = int(w)
    blur_height = int(h)
    roi = img[int(by):int(ty), int(tx):int(bx)]
    blur_image = cv2.GaussianBlur(roi,(85,85),0)
    img[int(by):int(ty), int(tx):int(bx)] = blur_image
    cv2.imwrite(save_path + name, img)

def blur_bbox(file, annotations, save_path):
    """Blurs the bounding boxes of the image.
    
    Args:
        img: The image to blur.
        annotations: bounding box coordinates
        save_path: path to save the image
        
    Returns:
        The image with the bounding boxes blurred.
    """
    bbox = []
    temp = file.split('/')
    name = temp[-1]
    img = cv2.imread(file)
    for i in annotations:
        ll = float(i) * 416
        bbox. append(int(ll))
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    tx = x - (w/2) 
    ty = y + (h/2) 
    bx = x + (w/2) 
    by = y - (h/2) 
    blur_x = int(tx)
    blur_y = int(ty)
    blur_width = int(w)
    blur_height = int(h)
    roi = img[int(by):int(ty), int(tx):int(bx)]
    blur_image = cv2.GaussianBlur(roi,(85,85),0)
    img[int(by):int(ty), int(tx):int(bx)] = blur_image
    cv2.imwrite(save_path + name, img)


def blur_class(input_folder, output_folder):
    try:
        shutil.copy(input_folder + "classes.txt", output_folder)
    except:
        pass
    try:
        shutil.copy(input_folder + "_darknet.labels", output_folder)
    except:
        pass
    for file in os.listdir(input_folder):
        if file.endswith(".jpg"):
            if os.path.isfile(input_folder + file[:-4] + '.txt'):
                annot = open(input_folder + file[:-4]+'.txt')
                for line in annot:
                    li = line.split(' ')
                    if li[0] == '0':
                        annotation = [li[1], li[2], li[3], li[4][:-2]]
                        try:
                            blur_bbox_etended(input_folder + file, annotation, output_folder)
                        except:    
                            blur_bbox(input_folder + file, annotation, output_folder)
                        try:
                            shutil.copy(input_folder + file[:-4]+'.txt', output_folder)
                        except:
                            pass
                    elif li[0] == '1':
                        annotation = [li[1], li[2], li[3], li[4]]
                        try:
                            blur_bbox_etended(input_folder + file, annotation, output_folder)
                        except:    
                            blur_bbox(input_folder + file, annotation, output_folder)
                        try:
                            shutil.copy(input_folder + file[:-4]+'.txt', output_folder)
                        except:
                            pass
                    elif li[0] == '2':
                        annotation = [li[1], li[2], li[3], li[4]]
                        try:
                            blur_bbox_etended(input_folder + file, annotation, output_folder)
                        except:    
                            blur_bbox(input_folder + file, annotation, output_folder)
                        try:
                            shutil.copy(input_folder + file[:-4]+'.txt', output_folder)
                        except:
                            pass
                    elif li[0] == '3':
                        annotation = [li[1], li[2], li[3], li[4]]
                        try:
                            blur_bbox_etended(input_folder + file, annotation, output_folder)
                        except:    
                            blur_bbox(input_folder + file, annotation, output_folder)
                        try:
                            shutil.copy(input_folder + file[:-4]+'.txt', output_folder)
                        except:
                            pass
                    elif li[0] == '4':
                        annotation = [li[1], li[2], li[3], li[4]]
                        try:
                            blur_bbox_etended(input_folder + file, annotation, output_folder)
                        except:    
                            blur_bbox(input_folder + file, annotation, output_folder)
                        try:
                            shutil.copy(input_folder + file[:-4]+'.txt', output_folder)
                        except:
                            pass
            else:
                pass
        else:
            pass

def blur_out_x_class(input_folder, output_folder):
    try:
        shutil.copy(input_folder + "classes.txt", output_folder)
    except:
        pass
    try:
        shutil.copy(input_folder + "_darknet.labels", output_folder)
    except:
        pass
    for file in os.listdir(input_folder):
        if file.endswith(".jpg"):
            if os.path.isfile(input_folder + file[:-4] + '.txt'):
                annot = open(input_folder + file[:-4]+'.txt')
                for line in annot:
                    li = line.split(' ')
                    if li[0] == '0':
                        annotation = [li[1], li[2], li[3], li[4][:-2]]
                        try:
                            try:
                                blur_bbox_etended(output_folder + file, annotation, output_folder)
                            except:
                                blur_bbox(output_folder + file, annotation, output_folder)    
                        except:
                            try:
                                blur_bbox_etended(input_folder + file, annotation, output_folder)
                            except:    
                                blur_bbox(input_folder + file, annotation, output_folder)
                    elif li[0] == '1':
                        annotation = [li[1], li[2], li[3], li[4]]
                        try:
                            try:
                                blur_bbox_etended(output_folder + file, annotation, output_folder)
                            except:
                                blur_bbox(output_folder + file, annotation, output_folder)    
                        except:
                            try:
                                blur_bbox_etended(input_folder + file, annotation, output_folder)
                            except:    
                                blur_bbox(input_folder + file, annotation, output_folder)
                    elif li[0] == '2':
                        annotation = [li[1], li[2], li[3], li[4]]
                        try:
                            try:
                                blur_bbox_etended(output_folder + file, annotation, output_folder)
                            except:
                                blur_bbox(output_folder + file, annotation, output_folder)    
                        except:
                            try:
                                blur_bbox_etended(input_folder + file, annotation, output_folder)
                            except:    
                                blur_bbox(input_folder + file, annotation, output_folder)
                    elif li[0] == '3':
                        annotation = [li[1], li[2], li[3], li[4]]
                        cent_x = int(float(li[1]) * 416)
                        cent_y = int(float(li[2]) * 416)
                        width = int(float(li[3]) * 416)
                        height = int(float(li[4][:-2]) * 416)
                        x1 = int(cent_x - (width / 2))
                        y1 = int(cent_y - (height / 2))
                        x2 = int(cent_x + (width / 2))
                        y2 = int(cent_y + (height / 2))
                        if x1 <= 5 or y1 <= 5:
                            pass
                        elif x2 >= 411 or y2 >= 411:
                            pass
                        else:
                                with open(output_folder + file[:-4] + '.txt', 'a') as f:
                                    f.write(str(li[0]))
                                    f.write(' ')
                                    f.write(str(li[1]))
                                    f.write(' ')
                                    f.write(str(li[2]))
                                    f.write(' ')
                                    f.write(str(li[3]))
                                    f.write(' ')
                                    f.write(str(li[4]))
                                    f.write('\n')
                    elif li[0] == '4':
                        annotation = [li[1], li[2], li[3], li[4]]
                        try:
                            try:
                                blur_bbox_etended(output_folder + file, annotation, output_folder)
                            except:
                                blur_bbox(output_folder + file, annotation, output_folder)    
                        except:
                            try:
                                blur_bbox_etended(input_folder + file, annotation, output_folder)
                            except:    
                                blur_bbox(input_folder + file, annotation, output_folder)
            else:
                pass
        else:
            pass