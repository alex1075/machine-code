import cv2
import os
import decimal
import tqdm
import subprocess
import inquirer
import numpy as np
from code.helper.config import *
from code.helper.annotations import *
from code.helper.imageTools import *


host_file = os.getcwd() + '/code/data/hosts'

def check_full_path(path):
    if os.path.isabs(path) == True:
        return path
    else:
        if os.path.isabs(os.getcwd() + '/' + path) == True:
            return os.getcwd() + '/' + path
        else:
            raise Exception('Path not found')

def save_new_host():
    host = input('Enter hostname or IP: ')
    user = input('Enter username: ')
    with open(check_full_path(host_file), 'a') as f:
        f.write(host + ' ' + user + '\n')

def choose_host():
    hosts = []
    with open(check_full_path(host_file), 'r') as f:
        for line in f:
            hosts.append(line.strip())
    hosts.append('Add new host')        
    questions = [inquirer.List('host', message='Choose host', choices=hosts)]
    answers = inquirer.prompt(questions)
    if answers['host'] == 'Add new host':
        save_new_host()
        return choose_host()
    else:
        return answers['host']

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
    select_yolo_version(path + 'obj.names', path, 10)
    make_obj_data(path, False)
    
def get_file_remote(name, ssh):
    array = []
    name = '*'+name+'*'    
    proc = subprocess.run(["ssh", ssh, "-t", "find" , "/", "-name", name, '-print', ' 2>/dev/null'], stdout=subprocess.PIPE)
    out = str(proc.stdout.decode('utf-8'))
    out = out.split('\n')
    for line in out:
        li = line.strip('\r')
        if ': Permission denied' in str(li):
            pass
        elif 'Connection to' in str(li):
            pass
        else:
            if len(li) != 0:
                array.append(li)
            else:
                pass    
    return array

def get_file_local(name):
    array = []
    name = '*'+name+'*'
    proc = subprocess.run(["find" , "/", "-name", name, '-print', ' 2>/dev/null'], stdout=subprocess.PIPE)
    out = str(proc.stdout.decode('utf-8'))
    out = out.split('\n')
    for line in out:
        li = line.strip('\r')
        if ': Permission denied' in str(li):
            pass
        elif 'Connection to' in str(li):
            pass
        else:
            if len(li) != 0:
                array.append(li)
            else:
                pass    
    return array

def get_file_over(dest, name):
    loc = input('Local or Remote? (l/r): ')
    if loc == 'l':
        local = True
        array = get_file_local(name)
    elif loc == 'r':
        local = False
        host = choose_host()
        array = get_file_remote(name, host)
    if os.path.exists(dest) == False:
        os.mkdir(dest)
    dest = check_full_path(dest)
    question = [inquirer.List('file',
                           message="Which file do you want to copy?",
                           choices=array,
                       ),]
    answer = inquirer.prompt(question)
    print(answer['file'])
    if local == True:
        os.system('cp ' + answer['file'] + ' ' + dest)
    elif local == False:
        os.system('scp ' + host + ':' + answer['file'] + ' ' + dest)

def choose_folder(path):
    array = []
    os.listdir(path)
    for d in os.listdir(path):
        if os.path.isdir(path + d) == True:
            array.append(path + d)
    question = [inquirer.List('folder',
                            message="Which AI model folder do you want to use?",
                            choices=array,
                        ),]
    answer = inquirer.prompt(question)
    path = answer['folder'] + '/'
    return path  

def choose_weights(path):
    array = []
    path = check_full_path(path + 'backup/')
    for d in os.listdir(path):
        if d.endswith('.weights') == True:
            array.append(path + d)
    question = [inquirer.List('weights',
                            message="Which AI model weights do you want to use?",
                            choices=array,
                        ),]
    answer = inquirer.prompt(question)
    return answer['weights']  

def choose_cfg(path):
    array = []
    path = check_full_path(path)
    for d in os.listdir(path):
        if d.endswith('.cfg') == True:
            array.append(path + d)
    array.append('Other')        
    question = [inquirer.List('weights',
                            message="Which AI model weights do you want to use?",
                            choices=array,
                        ),]
    answer = inquirer.prompt(question)
    if answer['weights'] == 'Other':
        cfg = input('Enter path to cfg file: ')
        return cfg
    else:
        return answer['weights']
    

def cat_file(file):
    os.system('cat ' + file)

def parent_dir(path):
    return os.path.abspath(os.path.join(path, os.pardir)) + '/'

def video_len(filename):
    import cv2
    video = cv2.VideoCapture(filename)
    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    return duration, frame_count    

def select_yolo_version(obj_names, path, upper_range):
    question = [inquirer.List('version',
                            message="Which YOLO version do you want to use?",
                            choices=['yolov4', 'yolov5', 'yolov6', 'yolov7', 'yolov8'],
                        ),]
    answer = inquirer.prompt(question)
    if answer['version'] == 'yolov4':
        prepare_cfg_v4(obj_names, path, upper_range)
    elif answer['version'] == 'yolov5':
        prepare_cfg_v5(obj_names, path, upper_range)
    else:
        raise Exception('Not yet implemented')


def choose_epochs():
    question = [inquirer.List('epochs',
                            message="How many epochs do you want to train?",
                            choices=['50', '100', '250', '500', '750', '1000', '5000', '10000', 'Other'],
                        ),]
    answer = inquirer.prompt(question)
    if answer['epochs'] == 'Other':
        epochs = input('Enter number of epochs: ')
        return epochs
    else:
        return answer['epochs']
    
def yes_no_question(question):
    question = [inquirer.List('yesno',
                            message=question,
                            choices=['Yes', 'No'],
                        ),]
    answer = inquirer.prompt(question)
    if answer['yesno'] == 'Yes':
        return 'y'
    else:
        return 'n'
