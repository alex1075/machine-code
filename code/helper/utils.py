import cv2
import os
import decimal
import tqdm
from code.helper.config import *

def check_full_path(path):
    if os.path.isabs(path) == True:
        return path
    else:
        if os.path.isabs(os.getcwd() + '/' + path) == True:
            return os.getcwd() + '/' + path
        else:
            raise Exception('Path not found')


#Grabs biggest dimension and scales the photo so that max dim is now 1280
def resizeTo(image, newhigh=1280, newwid=1280, inter=cv2.INTER_AREA):
    (height, width) = image.shape[:2]
    if height>width:
        newheight = newhigh
        heightratio = height/newheight
        newwidth = int(width/heightratio)
        resized = cv2.resize(image, dsize=(newwidth, newheight), interpolation=inter)
        return resized, newheight, newwidth
    elif width>height:
        newwidth = newwid
        widthratio = width/newwidth
        newheight = int(height/widthratio)
        resized = cv2.resize(image, dsize=(newwidth, newheight), interpolation=inter)
        return resized, newheight, newwidth
    else: 
        pass

def remove_non_annotated(pathtofolder):
    images = os.listdir(pathtofolder)
    for image in images:    
        if image.endswith(".jpg"):
            # check file exists
            if os.path.isfile(pathtofolder + image[:-4] + '.txt'):
               pass
            else:
                os.remove(pathtofolder + image[:-4] + '.jpg')
    
def remove_empty_lines(filename):
    if not os.path.isfile(filename):
        return
    with open(filename) as filehandle:
        lines = filehandle.readlines()
    with open(filename, 'w') as filehandle:
        lines = filter(lambda x: x.strip(), lines)
        filehandle.writelines(lines) 

def make_ground_truth_file(path, gtfile):
    gt_file = open(gtfile, 'w')
    for file in os.listdir(path):
        if file.endswith('.txt'):
            if file == 'test.txt':
                pass
            elif file == 'train.txt':
                pass
            elif file == 'valid.txt':
                pass
            else:
                img_name = file[:-4] + '.jpg'
                count = 0
                annot = open(path + file, 'r')
                for line in annot:
                    gt_file.write(img_name[:-4] + ' ' + line + '\n')
                    count += 1
                annot.close()
    remove_empty_lines(gtfile)
    gt_file.close()

def save_arrya_to_csv(array, path, file):
    f = open(file, 'w')
    for item in array:
        f.write("%s,%s,%s,%s,%s,%s\n" % (item[0], item[1], item[2], item[3], item[4], item[5]))
    f.close()

def split_img_label(data_train,data_test,folder_train,folder_test):
    try:
        os.mkdir(folder_train)
    except:
        pass
    try:
        os.mkdir(folder_test)
    except:
        pass
    train_ind=list(data_train.index)
    test_ind=list(data_test.index)
    # Train folder
    for i in tqdm.tqdm(range(len(train_ind))):
        os.system('cp '+data_train[train_ind[i]]+' ./'+ folder_train + '/'  +data_train[train_ind[i]].split('/')[2])
        os.system('cp '+data_train[train_ind[i]].split('.jpg')[0]+'.txt'+'  ./'+ folder_train + '/'  +data_train[train_ind[i]].split('/')[2].split('.jpg')[0]+'.txt')
    # Test folder
    for j in tqdm.tqdm(range(len(test_ind))):
        os.system('cp '+data_test[test_ind[j]]+' ./'+ folder_test + '/'  +data_test[test_ind[j]].split('/')[2])
        os.system('cp '+data_test[test_ind[j]].split('.jpg')[0]+'.txt'+'  ./'+ folder_test + '/'  +data_test[test_ind[j]].split('/')[2].split('.jpg')[0]+'.txt')  

def import_images_from_roboflow(url, path):
    os.system('curl -L -s '+url+' > roboflow.zip')
    os.system('unzip -qq roboflow.zip')
    os.system('rm roboflow.zip')
    if os.path.exists('train/') == True:
        os.system('mv train '+path)
        os.system('mv ' + path + 'train/_darknet.labels '+path)
    if os.path.exists('test/') == True:
        os.system('mv test '+path)
        if os.path.exists(path+'train/') == True:
            pass
        else:
            os.system('mv ' + path + 'test/_darknet.labels '+path)
    if os.path.exists('valid/') == True:
        os.system('mv valid '+path)
        if os.path.exists(path+'train/') == True:
            if os.path.exists(path+'test/') == True:
                pass
            else:
                pass
        else:
            os.system('mv ' + path + 'valid/_darknet.labels '+path)
    import_names(path, False)
    try:
        os.system('rm README.roboflow.txt')
    except:
        pass

def prepare_training(url, path):
    if os.path.exists(path) == True:
        raise Exception('Path already exists')
    else:
        os.mkdir(path)
    import_images_from_roboflow(url, path)
    if os.path.exists(path+'train/') == True:
        remove_non_annotated(path + 'train/')
        prep(path + 'train/', 'train.txt')
    if os.path.exists(path+'test/') == True:
        remove_non_annotated(path + 'test/')
        prep(path + 'test/', 'test.txt')
    if os.path.exists(path+'valid/') == True:
        remove_non_annotated(path + 'valid/')
        prep(path + 'valid/', 'valid.txt')
    prepare_cfg('code/data/yolov4.cfg', path + 'obj.names', path, 10, 'yolov4_10.cfg')
    make_obj_data(path, False)
    