import cv2
import glob, os
from PIL import Image
import numpy as np
import random
import shutil
import decimal
import socket
import tqdm


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
                                f.write(str(classes))
                                f.write(' ')
                                f.write(str(round(x1, 6)))
                                f.write(' ')
                                f.write(str(round(y1, 6)))
                                f.write(' ')
                                f.write(str(round(x2, 6)))
                                f.write(' ')
                                f.write(str(round(y2, 6)))
                                f.write('\n')
                else:
                    pass
        else:
            pass
    
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

def send_file(path, file, server, port):
    """
    Send a file to a server.
    """
    # get the file size
    filesize = os.path.getsize(path + file)
    filename = file
    SEPARATOR = "<SEPARATOR>"
    BUFFER_SIZE = 4096 # send 4096 bytes each time step
    # create the client socket
    s = socket.socket()
    print(f"[+] Connecting to {server}:{port}")
    
    # connect to the server
    s.connect((server, port))
    print("[+] Connected.")
    # send the filename and filesize
    # start sending the file
    progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "rb") as f:
        while True:
            # read the bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                # file transmitting is done
                break
            # we use sendall to assure transimission in 
            # busy networks
            s.sendall(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))
    # close the socket
    s.close()
    print(f"[+] File {file} sent to {server}:{port}")

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
