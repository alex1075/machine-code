from turtle import st
import cv2
import re
import glob, os, datetime
from PIL import Image
from imutils import paths
from code.helper.utils import *
from code.helper.imageTools import *
from code.helper.annotations import *
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


def convert(path_to_folder='/Volumes/PhD/PhD/Data/'):
    for infile in os.listdir(path_to_folder):
        print ("file : " + infile)
        if infile[-3:] == "bmp":
            print ("is bmp")
            outfile = infile[:-3] + "jpg"
            im = Image.open(path_to_folder + infile)
            print ("new filename : " + outfile)
            out = im.convert("RGB")
            out.save(path_to_folder + outfile, "jpeg", quality=100)
            os.remove(path_to_folder + infile)
        elif infile[-4:] == "tiff":
            print ("is tiff")
            outfile = infile[:-4] + "jpg"
            im = Image.open(path_to_folder + infile)
            out = im.convert("RGB")
            out.save(path_to_folder + outfile, "jpeg", quality=100)
            os.remove(path_to_folder + infile)
        elif infile[-3:] == "png":
            print ("is png")
            outfile = infile[:-3] + "jpg"
            img = cv2.imread(path_to_folder + infile)
            print(path_to_folder + outfile)
            cv2.imwrite(path_to_folder + outfile, img)
            os.remove(path_to_folder + infile)
        elif infile[-3:] == "jpg" or infile[-3:] == "jpeg":
            print ("is jpg, no change")
        else:
            print ("Not an image")

#Cycles through iamges in path_to_folder and resize them the desired size
def resizeAllJpg(path_to_folder='/Volumes/PhD/PhD/Data/', newhight=1080, newwid=1080):
  jpgs = glob.glob(path_to_folder + '*.jpg')
  for image in jpgs:
      print ("resizing image" + image)
      name_without_extension = os.path.splitext(image)[0]
      img = cv2.imread(image)
      resized, newheight, newwidth = resizeTo(img, newhight, newwid)
      cv2.imwrite(name_without_extension + ".jpg", resized)

#Cycles through videos in path_to_folder and outputs jpg to out_folder
def convertVideoToImage(path_to_folder='/Volumes/PhD/PhD/Video/', out_folder='/Volumes/PhD/PhD/Data/'):
    for fi in os.listdir(path_to_folder):
        nam, ext = os.path.splitext(fi)
        if ext == '.mp4':
            cam = cv2.VideoCapture(path_to_folder + fi)
            try:
                # creating a folder named data
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)
            # if not created then raise error
            except OSError:
                print ('Error: Creating directory of' + out_folder)
            # frame
            currentframe = 0
            while(True):
                # reading from frame
                ret,frame = cam.read()
                if ret:
                    # if video is still left continue creating images
                    name = out_folder + nam + '_frame_' + str(currentframe) + '.jpg'
                    print ('Creating...' + name)
                    # writing the extracted images
                    cv2.imwrite(name, frame)
                    # increasing counter so that it will
                    # show how many frames are created
                    currentframe += 1
                else:
                    break
            # Release all space and windows once done
            cam.release()
            cv2.destroyAllWindows()

def normalise(path_to_folder=r'/Volumes/PhD/PhD/Data/'):
    jpgs = glob.glob(path_to_folder + '*.jpg')
    for infile in jpgs:
        print ("file : " + infile)
        flute = cv2.imread(infile)
        print ("Normalising " + infile)
        im = normaliseImg(flute)
        cv2.imwrite(path_to_folder + infile, im)

def randomCrop(path_to_folder='/Volumes/PhD/PhD/Data/', outfolder='/Volumes/PhD/PhD/Dataset/', crop_height=256, crop_width=256):
    jpgs = glob.glob(path_to_folder + '*.jpg')
    for jpg in jpgs:
        print("randomly cropping: " + jpg)
        flute = cv2.imread(jpg, 0)
        jpg_name = jpg.split('/')[-1]
        infile = jpg_name[:-4] + "_cropped_" + str(datetime.datetime.now()) + ".jpg"
        print(infile)
        im = getRandomCrop(flute, crop_height, crop_width)
        cv2.imwrite(outfolder + infile, im)

def convert2Gray(path_to_folder='/Volumes/PhD/PhD/Dataset/'):
    jpgs = glob.glob(path_to_folder  + '*.jpg')
    for jpg in jpgs:
        print('Converting to grayscale: ' + jpg)
        flute = cv2.imread(jpg, 0)
        print(flute.shape)
        cv2.imwrite(jpg, flute)

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def detectBlurr(path_to_folder='/Volumes/PHD/', threshold=100.0):
    file = open(path_to_folder + 'recap.txt', "w")
    file.write('Threshold: ' + str(threshold) + '\n')
    print('Detecting blurr')
   # loop over the input images
    for imagePath in paths.list_images(path_to_folder):
        # load the image, convert it to grayscale, and compute the
        # focus measure of the image using the Variance of Laplacian
        # method
        print("Checking image: " + imagePath)
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        text = 'Not Blurry'
 
        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry"
        if fm < threshold:
            text = 'Blurry'
            print(imagePath)
            # file.write(fm + "<" + threshold + '\n')
            file.write('Below threshold' + imagePath + '\n')
        # print(image)
    
    file.close()

def iterateBlur(path_to_folder='/Volumes/PHD/', start=0, end=100, step=5):
    for i in range(start, end, step):
        print('Threshold: ' + str(i))
        detectBlurr(path_to_folder, threshold=i)

def detectAndMoveBlurr(path_to_folder='/Volumes/PHD/', threshold=100.0, currentstep=110, outfolder='/Volumes/PHD/sorted/'):
    file = open(outfolder + 'recap.txt', "w")
    file.write('Threshold: ' + str(threshold) + '\n')
    print('Detecting blurr')
    os.makedirs(outfolder + 'threshold_' + str(threshold), exist_ok=True)
   # loop over the input images
    for imagePath in paths.list_images(path_to_folder):
        # load the image, convert it to grayscale, and compute the
        # focus measure of the image using the Variance of Laplacian
        # method
        print("Checking image: " + imagePath)
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry"
        if fm < threshold & threshold < currentstep:
            print(imagePath)
            # file.write(fm + "<" + threshold + '\n')
            file.write('Below threshold' + imagePath + '\n')
            shutil.copy(imagePath, outfolder + 'threshold_' + str(threshold))
            os.remove(imagePath)
            print("Moved " + imagePath)
            file.write('Moved ' + imagePath + ' to folder' + '\n')

    file.close()

def iterateBlurMove(path_to_folder='/Volumes/PHD/', outfolder='/Volumes/PHD/sorted/', start=0, end=100, step=5):
    for i in range(start, end, step):
        print('Threshold: ' + str(i))
        currentstep = str(range) + str(step)
        detectAndMoveBlurr(path_to_folder, threshold=i, currentstep=currentstep, outfolder=outfolder)

def selectIMG(path_to_folder='/Volumes/PHD/sorted/', outfolder='/Volumes/PHD/sorted/', number=200):
    randomSelect(path_to_folder, outfolder, number)

def chopUpDataset(path_to_folder='/Users/alexanderhunt/Preprocessing/test_dataset/', outfolder='/Users/alexanderhunt/Preprocessing/output/', x=416, y=416, annotations=True):
    crop_images(x, y, path_to_folder, outfolder, annotations)
    if annotations == True:
        remove_non_annotated(outfolder)
    else:
        pass
    checkAllImg(outfolder, x, y)

# def batchBackgroundRemove(path_to_folder='output/', background_folder='background/', outfolder='/Volumes/PHD/removed/'):
#     return print('Not implemented yet')

def batchBackgroundRemove(path_to_folder='output/', background_folder='backgrounds/', outfolder='data_4/', alpha=2):
    list_img=[img for img in os.listdir(path_to_folder) if img.endswith('.jpg')==True]
    list_background=[img for img in os.listdir(background_folder) if img.endswith('.jpg')==True]
    list_txt=[img for img in os.listdir(path_to_folder) if img.endswith('.txt')==True]
    for img in list_img:
        index = img.split('_')
        name = index[0] + '_' + index[1]
        first_chop = index[2]
        second_chop = index[3]
        condition = 'background''_' + str(first_chop) + '_' + str(second_chop)
        img_path = outfolder + img
        background_element = [x for x in list_background if x==condition]
        background = str(background_element[0])
        img = background_removal_with_alpha(path_to_folder+img, background_folder+background, alpha)
        cv2.imwrite(img_path, img)
        print('Matched ' + img_path + ' with ' + background)
        annotation_condition  = name + '_' + str(first_chop) + '_' + str(second_chop)[:-4] + '.txt'
        annotation_file = [x for x in list_txt if x==annotation_condition]
        annotation = str(annotation_file[0])
        print('Matched ' + annotation + ' with ' + annotation_condition)
        os.system('cp ' + path_to_folder + annotation + ' ' + outfolder + annotation)
    classes_file = [x for x in list_txt if x=='classes.txt']
    classes = str(classes_file[0])
    os.system('cp ' + path_to_folder + classes + ' ' + outfolder + classes)


def splitDataset(path_to_folder, outfolder):
    list_img=[img for img in os.listdir(path_to_folder) if img.endswith('.jpg')==True]
    list_txt=[img for img in os.listdir(path_to_folder) if img.endswith('.txt')==True]
    path_img=[]
    for i in range (len(list_img)):
        path_img.append(path_to_folder+list_img[i])
    df=pd.DataFrame(path_img)
    # split 
    data_train, data_test, labels_train, labels_test = train_test_split(df[0], df.index, test_size=0.20, random_state=42)
    # Function split 
    split_img_label(data_train, data_test, outfolder+'train/', outfolder+'test/')
    os.system('cp' + ' ./'+ path_to_folder + '/'  +'classes.txt' + ' ./'+ outfolder + 'train/')
    os.system('cp' + ' ./'+ path_to_folder + '/'  +'classes.txt' + ' ./'+ outfolder + 'test/')

def import_results(input_file='result.txt', results_file='results.txt'):
    res = open(results_file, 'w')
    with open(input_file, 'r') as f:
        for line in f:
            if line[0:4] == '/hom':
                lin = re.split('/| ', line)
                li = filter(lambda a: '.jpg' in a, lin)
                l = list(li)[0][:-5]
                print(l)
                image_name = l
            elif line[0:4] == 'ERY:':
                lin = re.split(':|%|t|w|h', line)
                # print(lin)
                classes = 1
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
                # print(res)
                res.write(image_name + ' ' + str(classes) + ' ' + str(left_x) + ' ' + str(top_y) + ' ' + str(width) + ' ' + str(height) + ' ' + str(confidence / 100) + ' \n')
            elif line[0:4] == 'ECHY':
                lin = re.split(':|%|t|w|h', line)
                # print(lin)
                classes = 0
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
                res.write(image_name + ' ' + str(classes) + ' ' + str(left_x) + ' ' + str(top_y) + ' ' + str(width) + ' ' + str(height) + ' ' + str(confidence / 100) + ' \n')
                # print(res)
            else:
                pass


import os
def get_info(data_path, model_path, model_name):
    cfg = model_path + 'yolov4_10.cfg'
    weights = model_path + 'backup/' + model_name
    data = model_path + 'obj.data'
    temp_path = data_path + 'temp/'
    if os.path.exists(temp_path) == True:
        pass
    else:
        os.mkdir(temp_path)
    if os.path.exists(data_path + 'test.txt') == True:
        os.remove(data_path + 'test.txt')
    else:
        filoo = open(data_path + 'test.txt', 'w')
        for image in os.listdir(data_path):
            if image.endswith(".jpg"):
                # print(image)
                filoo.write(data_path + image + "\n")
        filoo.close()
    os.system('darknet detector test ' + data + ' ' + cfg + ' ' + weights + ' -dont_show -ext_output < ' + data_path + 'test.txt' + ' > ' + temp_path + 'result.txt 2>&1')
    results = open(temp_path + 'result.txt', 'r')
    lines = results.readlines()
    # print(lines)
    save = []
    cells = ('LYM:', 'MON:', 'NEU:', 'ERY:', 'PLT:', 'ECHY', 'WBC:')
    for line in lines:
        if line[0:4] in cells:
            # print(line)
            # print(line[0], line[1])
            lin = re.split(':|%|t|w|h', line)
            save.append([lin[0], int(lin[1])])
        else:
            pass    
    df = pd.DataFrame(save, columns=['Cell type', 'Confidence'])    
    # print(df)    
    os.remove(data_path + 'test.txt')
    df.to_csv(data_path + 'results.csv', index=False)
    ery = df.loc[df['Cell type'] == 'ERY']
    echy = df.loc[df['Cell type'] == 'ECHY']
    plt = df.loc[df['Cell type'] == 'PLT']
    wbc = df.loc[df['Cell type'] == 'WBC']
    lym = df.loc[df['Cell type'] == 'LYM']
    mon = df.loc[df['Cell type'] == 'MON']
    neu = df.loc[df['Cell type'] == 'NEU']
    print('Counted ' + str(len(df)) + ' cells')
    print('Overall average confidence: ' + str(round(float(df['Confidence'].mean()), 2)))
    if len(ery) != 0:
        print('Counted ' + str(len(ery)) + ' erythrocytes')
        print('Average confidence: ' + str(round(float(ery['Confidence'].mean()), 2)))
    if len(echy) != 0:
        print('Counted ' + str(len(echy)) + ' echinocytes')
        print('Average confidence: ' + str(round(float(echy['Confidence'].mean()), 2)))
    if len(plt) != 0:
        print('Counted ' + str(len(plt)) + ' platelets')
        print('Average confidence: ' + str(round(float(plt['Confidence'].mean()), 2)))
    if len(wbc) != 0:
        print('Counted ' + str(len(wbc)) + ' white blood cells')
        print('Average confidence: ' + str(round(float(wbc['Confidence'].mean()), 2)))
    if len(lym) != 0:
        print('Counted ' + str(len(lym)) + ' lymphocytes')
        print('Average confidence: ' + str(round(float(lym['Confidence'].mean()), 2)))
    if len(mon) != 0:
        print('Counted ' + str(len(mon)) + ' monocytes')
        print('Average confidence: ' + str(round(float(mon['Confidence'].mean()), 2)))
    if len(neu) != 0:
        print('Counted ' + str(len(neu)) + ' neutrophils')
        print('Average confidence: ' + str(round(float(neu['Confidence'].mean()), 2)))
    if len(wbc) != 0:
        import_and_filter_results(temp_path + 'result.txt', temp_path + 'results.txt')
    else:
        import_and_filter_result_2(temp_path + 'result.txt', temp_path + 'results.txt')   
    with open(temp_path + 'results.txt') as f:
        for line in f:
            item = line.split()
            mv = [float(item[2]), float(item[3]), float(item[4]), float(item[5])]
            mv = [i / 416 for i in mv]
            with open(temp_path + item[0] + '.txt', 'a') as g:
                 g.write(str(item[1]) + ' ' + str(mv[0]) + ' ' + str(mv[1]) + ' ' + str(mv[2]) + ' ' + str(mv[3]) + '\n')
    os.remove(temp_path + 'results.txt')           
    os.remove(temp_path + 'result.txt')