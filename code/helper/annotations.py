import os
import re
import cv2
import decimal

def add_bbox(image, bbox, classes):
    """
    Adds a coloured bounding box to an image. Colour is defined as the class passed to the function.

    Classes:
    0 - Red
    1 - Green
    2 - Blue
    3 - Purple
    4 - Yellow
    5 - Cyan
    6 - Orange
    """
    classes = int(classes)
    image = cv2.imread(image)
    if classes == 0:
        colour = (0,0,255) #red
    elif classes == 1:
        colour = (0,255,0) #green
    elif classes == 2:
        colour = (255,0,0) #blue
    elif classes == 3:
        colour = (255,0,255) #fuchsia 
    elif classes == 4:
        colour = (255,255,0) #yellow
    elif classes == 5:
        colour = (0,255,255) #cyan
    elif classes == 6:
        colour = (255,172,28) #orange
    elif classes == 7:
        colour = (255,255,255) #white
    elif classes == 8:
        colour = (0,0,0) #black
    elif classes == 9:
        colour = (235,92,135) #pink
    elif classes == 10:
        colour = (91,5,145) #purple
    elif classes == 11:
        colour = (173,241,33) #lime
    elif classes == 12:
        colour = (137,73,80) #brown 
    else:
        print('ERROR: We dont have a colour setup for that class')
    left_x = bbox[0]
    top_y = bbox[1]
    right_x = bbox[2]
    bottom_y = bbox[3]
    img = cv2.rectangle(image, (left_x,top_y), (right_x,bottom_y), colour, 2)
    return img

def change_annotation(i, j, x, y, height, width, path, image, save_name, save_path):
    # read annotation
    with open(path + image[:-4] + '.txt', 'r') as f:
        lines = f.readlines()
    # loop over lines
    for line in lines:
        # get line
        line = line.split(' ')
        # get coordinates
        classes = int(line[0])
        x1 = decimal.Decimal(line[1]) #centre x
        y1 = decimal.Decimal(line[2]) #centre y
        x2 = decimal.Decimal(line[3]) #width
        y2 = decimal.Decimal(line[4]) #height
        if int(x1 * width) in range(j, j + x, 1):
                if int(y1 * height) in range(i, i + y, 1):
                        # get new coordinates
                        x1 = decimal.Decimal(((x1 * width) - j ) / x)
                        y1 = decimal.Decimal(((y1 * height) - i) / y)
                        x2 = decimal.Decimal(str((x2 * width) / x))
                        y2 = decimal.Decimal(str((y2 * height) / y))
                        x1 = float(x1)
                        y1 = float(y1)
                        x2 = float(x2)
                        y2 = float(y2)
                        min = float(0.05)
                        max = float(0.95)
                        # write new coordinates
                        if x1 <= min or y1 <= min or x1 >=max or y1 >= max  or x2 <= min or y2 <= min or x2 >= max or y2 >= max:
                            pass
                        else:
                            with open(save_path + save_name + '.txt', 'a') as f:
                                f.write(str(classes) + ' ' + str(round(x1, 6)) + ' ' + str(round(y1, 6)) + ' ' + str(round(x2, 6)) + ' ' + str(round(y2, 6)) + '\n')
                else:
                    pass
        else:
            pass

def iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] <= bb1[2]

    assert bb1[1] <= bb1[3]
    assert bb2[0] <= bb2[2]
    assert bb2[1] <= bb2[3]


    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box.
    # NOTE: We MUST ALWAYS add +1 to calculate area when working in
    # screen coordinates, since 0,0 is the top left pixel, and w-1,h-1
    # is the bottom right pixel. If we DON'T add +1, the result is wrong.
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def iterate_over_images(list, path_to_images, save_directory, name):
    '''Iterates over a list of images and adds bounding boxes to them'''
    fill = open(list, 'r')
    for line in fill:
        lin = line.split(' ')
        image = lin[0]
        classes = str(lin[1])
        x1 = int(lin[2])
        y1 = int(lin[3])
        x2 = int(lin[4])
        y2 = int(lin[5])
        confidence = lin[6]
        if list == 'pd.txt':
            if classes == 0:
                classes == 4
            elif classes == 1:
                classes == 5
        elif list[0] == 'r':
            if classes == 0:
                classes == 4
            elif classes == 1:
                classes == 5

        bbox_coordinates = [x1, y1, x2, y2]  
        img = add_bbox(path_to_images + image + '.jpg', bbox_coordinates, int(classes))
        cv2.imwrite(save_directory + image + '_' + name + '.jpg', img)

def get_prediction_mistakes(gt_file, pd_file, path_to_images, save_directory):
    '''Compares the ground truth file to the prediction file and adds bounding
      boxes to the images where the prediction was incorrect'''
    gt = open(gt_file)
    pd = open(pd_file)
    for line in pd:
        li = line.split(' ')
        name = li[0]
        classes = li[1]
        bbox = [int(li[2]), int(li[3]), int(li[4]), int(li[5])]
        confidence = li[6]
        for lune in gt:
            lu = lune.split(' ')
            if lu[0] == name:
                nome = lu[0]
                clisses = lu[1]
                bbax = [int(lu[2]), int(lu[3]), int(lu[4]), int(lu[5])]
                canfidence = lu[6]
                if iou(bbox, bbax) >= 0.5:
                    if classes == clisses:
                        pass
                    else:
                        classes = 3
                        img = add_bbox(path_to_images + name+ '.jpg', bbox, int(classes))
                        cv2.imwrite(save_directory + name + '.jpg', img) 
                else:
                    if classes== 0:
                        classes = 4
                    elif classes == 1:
                        classes = 5
                    elif classes == 2:
                        classes = 6    
                    img = add_bbox(path_to_images + name+ '.jpg', bbox, int(classes))
                    cv2.imwrite(save_directory + name + '.jpg', img)        

def import_results_neo(input_file='result.txt', results_file='results.txt', obj_names='/home/as-hunt/Etra-Space/white-thirds/obj.names'):
    '''Import's Yolo darknet detection results and filters bounding 
    boxes that are outside of the image dimensions
    This function will use the index given to darknet 
    when training the model to determine the class of the object

    This function does not filter the result.txt file.
    
    Args:
        input_file (str): The path to the results file
        results_file (str): The path to the output file
        obj_names (str): The path to the obj.names file
        '''
    arry = []
    res = open(results_file, 'w')
    with open(obj_names, 'r') as f:
        for line in f:
            arry.append(line.rstrip())
    with open(input_file, 'r') as f:
        for line in f:
            if line[0:4] == '/hom':
                lin = re.split('/| ', line)
                li = filter(lambda a: '.jpg' in a, lin)
                l = list(li)[0][:-5]
                image_name = l
            elif (line[0:3] in arry) or (line[0:4] in arry ) == True:
                lin = re.split(':|%|t|w|h', line)
                classes = int(arry.index(lin[0]))
                confidence = int((lin[1]))
                if int(lin[4]) < 0:
                    left_x = 0
                else:
                    left_x = int(lin[4])
                if int(lin[6]) < 0:
                    top_y = 0
                else:
                    top_y = int(lin[6])
                width = int(lin[10])
                height = int(lin[14][:-2])
                bottom_y = top_y + height
                right_x = left_x + width
                res.write(image_name + ' ' + str(classes) + ' ' + str(left_x) + ' ' + str(top_y) + ' ' + str(right_x) + ' ' + str(bottom_y) + ' ' + str(confidence / 100) + ' \n')          
            else:
                pass 

def make_ground_truth(ground_truth_file='gt.txt', test_folder='valid/'):
    '''Creates a ground truth file from the annotations in a given folder'''
    gt_file = open(ground_truth_file, 'w')
    for file in os.listdir(test_folder):
        if file.endswith('.txt'):
            if file == 'test.txt':
                pass
            elif file == 'classes.txt':
                pass
            elif file == 'train.txt':
                pass
            elif file == 'valid.txt':
                pass
            elif file == 'ground_truth.txt':
                pass
            elif file == 'result_run_1.txt':
                pass
            elif file == 'result_run_2.txt':
                pass
            else:
                img_name = file[:-4] 
                count = 0
                annot = open(test_folder + file, 'r+')
                for line in annot:
                    lin = re.split(' ', line)
                    classes = lin[0]
                    center_x = lin[1]
                    center_y = lin[2]
                    width = lin[3]
                    height = lin[4]
                    if center_x == '0':
                        pass
                    elif center_y == '0':
                        pass
                    elif width == '0':
                        pass
                    elif height == '0':
                        pass
                    else:
                        center_x = decimal.Decimal(center_x) * 416
                        center_y = decimal.Decimal(center_y) * 416
                        width = decimal.Decimal(width) * 416
                        height = decimal.Decimal(height) * 416
                        left_x = int(decimal.Decimal(center_x) - (width / 2))
                        top_y = int(decimal.Decimal(center_y) - (height / 2))
                        right_x = int(decimal.Decimal(center_x) + (width / 2))
                        bottom_y = int(decimal.Decimal(center_y) + (height / 2))
                        if left_x <= 6:
                            pass
                        elif left_x >= 410:
                            pass
                        else:
                            if top_y <= 6:
                                pass
                            elif top_y >= 410:
                                pass
                            else:
                                if right_x <= 6:
                                    pass
                                elif right_x >= 410:
                                    pass
                                else:
                                    if bottom_y <= 6:
                                        pass
                                    elif bottom_y >= 410:
                                        pass
                                    else:
                                        gt_file.write(img_name + ' ' + str(classes) + ' ' + str(left_x) + ' ' + str(top_y) + ' ' + str(right_x) + ' ' + str(bottom_y) + ' \n')
                                        count += 1

def make_ground_truth_unfiltered(ground_truth_file='gt.txt', test_folder='valid/'):
    '''Creates a ground truth file from the annotations in a given folder, this function filters by the image dimensions limits'''
    gt_file = open(ground_truth_file, 'w')
    for file in os.listdir(test_folder):
        if file.endswith('.txt'):
            if file == 'test.txt':
                pass
            elif file == 'classes.txt':
                pass
            elif file == 'train.txt':
                pass
            elif file == 'valid.txt':
                pass
            elif file == 'ground_truth.txt':
                pass
            elif file == 'result_run_1.txt':
                pass
            elif file == 'result_run_2.txt':
                pass
            else:
                img_name = file[:-4] 
                count = 0
                annot = open(test_folder + file, 'r+')
                for line in annot:
                    lin = re.split(' ', line)
                    classes = lin[0]
                    center_x = lin[1]
                    center_y = lin[2]
                    width = lin[3]
                    height = lin[4]
                    if center_x == '0':
                        pass
                    elif center_y == '0':
                        pass
                    elif width == '0':
                        pass
                    elif height == '0':
                        pass
                    else:
                        center_x = decimal.Decimal(center_x) * 416
                        center_y = decimal.Decimal(center_y) * 416
                        width = decimal.Decimal(width) * 416
                        height = decimal.Decimal(height) * 416
                        left_x = int(decimal.Decimal(center_x) - (width / 2))
                        top_y = int(decimal.Decimal(center_y) + (height / 2))
                        right_x = int(decimal.Decimal(center_x) + (width / 2))
                        bottom_y = int(decimal.Decimal(center_y) - (height / 2))
                        gt_file.write(img_name + ' ' + str(classes) + ' ' + str(left_x) + ' ' + str(top_y) + ' ' + str(right_x) + ' ' + str(bottom_y) + ' \n')

def import_and_filter_result_neo(input_file='/home/as-hunt/result.txt', results_file='results.txt', obj_names='/home/as-hunt/Etra-Space/white-thirds/obj.names'):
    '''Import's Yolo darknet detection results bounding boxes.

    This function does filters the result.txt file. 
    It removes bounding boxes that are outside the image and
    bounding boxes that are too close to the edge of the image.

      Args:
        input_file (str): The path to the results.txt file
        results_file (str): The path to the file to save the filtered results
        obj_names (str): The path to the obj.names file
        '''
    arry = []
    res = open(results_file, 'w')
    with open(obj_names, 'r') as f:
        for line in f:
            arry.append(line.rstrip())
    with open(input_file, 'r') as f:
        for line in f:
            if line[0:4] == '/hom':
                lin = re.split('/| ', line)
                li = filter(lambda a: '.jpg' in a, lin)
                l = list(li)[0][:-5]
                image_name = l
            elif (line[0:3] in arry) or (line[0:4] in arry ) == True:
                lin = re.split(':|%|t|w|h', line)
                if int(lin[4]) < 4:
                    pass
                elif int(lin[4]) > 412:
                    pass
                else:
                    if int(lin[6]) < 4:
                        pass
                    elif int(lin[6]) > 412:
                        pass
                    else:
                        classes = int(arry.index(lin[0]))
                        confidence = int((lin[1]))
                        if int(lin[4]) < 0:
                            left_x = 0
                        else:
                            left_x = int(lin[4])
                        if int(lin[6]) < 0:
                            top_y = 0
                        else:
                            top_y = int(lin[6])
                        width = int(lin[10])
                        height = int(lin[14][:-2])
                        bottom_y = top_y + height
                        right_x = left_x + width
                        if bottom_y < 0:
                            bottom_y = 0
                        if right_x > 416:
                            right_x = 416
                        if bottom_y < 4:
                            pass
                        elif bottom_y > 412:
                            pass
                        else:
                            if right_x > 412:
                                pass
                            elif right_x < 4:
                                pass
                            else:
                                res.write(image_name + ' ' + str(classes) + ' ' + str(left_x) + ' ' + str(top_y) + ' ' + str(right_x) + ' ' + str(bottom_y) + ' ' + str(confidence / 100) + ' \n')
            else:
                pass                    

def check_all_annotations_for_duplicates(annotation_file):
    start = []
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            l = line.split(' ')
            image_name, classes, left_x, top_y, right_x, bottom_y, confidence = l[0], l[1], l[2], l[3], l[4], l[5], l[6]
            start.append([image_name, classes, left_x, top_y, right_x, bottom_y, confidence])
    for item in start:
        with open(annotation_file, 'a+') as f:
            lines = f.readlines()
            for line in lines:
                l = lines.split(' ')
                if item[0] == l[0]:
                    bbox_1 = [int(item[2]), int(item[3]), int(item[4]), int(item[5])]
                    bbox_2 = [int(l[2]), int(l[3]), int(l[4]), int(l[5])]
                    if iou(bbox_1, bbox_2) > 0.5:
                        if item[6] > l[6]:
                            lines.remove(line)
                        elif item[6] < l[6]:
                            start.remove(item)
                        elif item[6] == l[6]:
                            start.remove(item)

def del_edge_bbox_train(results_folder):
    for file in os.listdir(results_folder):
        if file.endswith('classes.txt'):
            pass
        elif file.endswith('.txt'):
            filoo = open(results_folder + file, 'r')
            lines = filoo.readlines()
            filoo.close()
            filoo = open(results_folder + file, 'w')
            for line in lines:
                lin = line.split(' ')
                if lin[0] == '\n':
                    pass
                else:
                    cent_x = float(lin[1])
                    cent_y = float(lin[2])
                    width = float(lin[3])
                    height = float(lin[4])
                    x1 = int(cent_x - (width / 2))
                    y1 = int(cent_y - (height / 2))
                    x2 = int(cent_x + (width / 2))
                    y2 = int(cent_y + (height / 2))

                    if x1 <= 5 or y1 <= 5 or x1 >= 411 or y1 >= 411:
                        pass
                    elif x2 <= 5 or y2 <= 5 or x2 >= 411 or y2 >= 411:
                        pass
                    else:
                        filoo.write(line)
            filoo.close()
            try:
                if os.stat(file).st_size == 0:
                    os.remove(file)
                    os.remove(file.replace('.txt', '.jpg'))
            except:
                pass
            try:
                if open(results_folder + file, 'r') == '\n':
                    os.remove(file)
                    os.remove(file.replace('.txt', '.jpg'))
            except:
                pass