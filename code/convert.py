import cv2
import re
import glob, os, time
from PIL import Image
from imutils import paths
from code.helper.utils import *
from code.helper.imageTools import *
from code.helper.annotations import *
import shutil
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split


def convert(path_to_folder='Data/', save_to_new=False, save_path='temp/'):
    for infile in tqdm.tqdm(os.listdir(path_to_folder), desc='Converting images', unit='images'):
        if infile[-3:] == "bmp":
            outfile = infile[:-3] + "jpg"
            im = Image.open(path_to_folder + infile)
            out = im.convert("RGB")
            if save_to_new == True:
                out.save(save_path + outfile, "jpeg", quality=100)
            else:
                out.save(path_to_folder + outfile, "jpeg", quality=100)
            os.remove(path_to_folder + infile)
        elif infile[-4:] == "tiff":
            outfile = infile[:-4] + "jpg"
            im = Image.open(path_to_folder + infile)
            out = im.convert("RGB")
            if save_to_new == True:
                out.save(save_path + outfile, "jpeg", quality=100)
            else:
                out.save(path_to_folder + outfile, "jpeg", quality=100)
            os.remove(path_to_folder + infile)
        elif infile[-3:] == "png":
            outfile = infile[:-3] + "jpg"
            img = cv2.imread(path_to_folder + infile)
            if save_to_new == True:
                cv2.imwrite(save_path + outfile, img)
            else:
                cv2.imwrite(path_to_folder + outfile, img)
            os.remove(path_to_folder + infile)
        elif infile[-3:] == "jpg" or infile[-3:] == "jpeg":
            try:
                shutil.copy(path_to_folder + infile, save_path + infile)
                os.remove(path_to_folder + infile)
            except:
                pass
        else:
            pass

#Cycles through iamges in path_to_folder and resize them the desired size
def resizeAllJpg(path_to_folder='Data/', newhight=1080, newwid=1080):
  jpgs = glob.glob(path_to_folder + '*.jpg')
  for image in jpgs:
      name_without_extension = os.path.splitext(image)[0]
      img = cv2.imread(image)
      resized, newheight, newwidth = resizeTo(img, newhight, newwid)
      cv2.imwrite(name_without_extension + ".jpg", resized)

#Cycles through videos in path_to_folder and outputs jpg to out_folder
def convertVideoToImage(path_to_folder='Video/', out_folder='Data/'):
    tic = time.perf_counter()
    for fi in os.listdir(path_to_folder):
        nam, ext = os.path.splitext(fi)
        if fi.endswith('.mp4'):
            cam = cv2.VideoCapture(path_to_folder + fi)
            all_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT)) 
            try:
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)
            except OSError:
                pass
            currentframe = 0
            with tqdm.tqdm(total=all_frames) as pbar:
                pbar.set_description('Converting video: ' + fi)
                while(True):
                    ret,frame = cam.read()
                    if ret:
                        name = out_folder + nam + '_frame_' + str(currentframe) + '.jpg'
                        cv2.imwrite(name, frame)
                        currentframe += 1
                        pbar.update(1)
                    else:
                        break
            pbar.close()        
            cam.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass
        else:
            pass
    toc = time.perf_counter()
    print(f"Finished in {toc - tic:0.4f} seconds")

def convertAVideoToImage(video, path):
            cam = cv2.VideoCapture(video)
            all_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT)) 
            currentframe = 0
            with tqdm.tqdm(total=all_frames) as pbar:
                pbar.set_description('Converting video: ' + video)
                while(True):
                    ret,frame = cam.read()
                    if ret:
                        name = video[:-4] + '_frame_' + str(currentframe) + '.jpg'
                        # print(name)
                        cv2.imwrite(name, frame)
                        currentframe += 1
                        pbar.update(1)
                    else:
                        break
            pbar.close()        
            cam.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass

def convert2Gray(path_to_folder='Dataset/'):
    jpgs = glob.glob(path_to_folder  + '*.jpg')
    for jpg in jpgs:
        flute = cv2.imread(jpg, 0)
        cv2.imwrite(jpg, flute)

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def detectBlurr(path_to_folder='', threshold=100.0):
    file = open(path_to_folder + 'recap.txt', "w")
    file.write('Threshold: ' + str(threshold) + '\n')
    for imagePath in paths.list_images(path_to_folder):
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        text = 'Not Blurry'
        if fm < threshold:
            text = 'Blurry'
            file.write('Below threshold' + imagePath + '\n')    
    file.close()

def iterateBlur(path_to_folder='', start=0, end=100, step=5):
    for i in range(start, end, step):
        print('Threshold: ' + str(i))
        detectBlurr(path_to_folder, threshold=i)

def detectAndMoveBlurr(path_to_folder='', threshold=100.0, currentstep=110, outfolder='sorted/'):
    file = open(outfolder + 'recap.txt', "w")
    file.write('Threshold: ' + str(threshold) + '\n')
    print('Detecting blurr')
    os.makedirs(outfolder + 'threshold_' + str(threshold), exist_ok=True)
    for imagePath in paths.list_images(path_to_folder):
        print("Checking image: " + imagePath)
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        if fm < threshold & threshold < currentstep:
            print(imagePath)
            file.write('Below threshold' + imagePath + '\n')
            shutil.copy(imagePath, outfolder + 'threshold_' + str(threshold))
            os.remove(imagePath)
            print("Moved " + imagePath)
            file.write('Moved ' + imagePath + ' to folder' + '\n')
    file.close()

def iterateBlurMove(path_to_folder='', outfolder='sorted/', start=0, end=100, step=5):
    for i in range(start, end, step):
        print('Threshold: ' + str(i))
        currentstep = str(range) + str(step)
        detectAndMoveBlurr(path_to_folder, threshold=i, currentstep=currentstep, outfolder=outfolder)

def chopUpDataset(path_to_folder='test_dataset/', outfolder='output/', x=416, y=416, annotations=True):
    tic = time.perf_counter()
    crop_images(x, y, path_to_folder, outfolder, annotations)
    toc = time.perf_counter()
    print(f"Finished in {toc - tic:0.4f} seconds")
    if annotations == True:
        remove_non_annotated(outfolder)
    else:
        pass
    checkAllImg(outfolder, x, y)
    del_top_n_bottom_parts(outfolder)

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

def check_for_img(path_to_folder):
    for image in os.listdir(path_to_folder):
        if image.endswith(".jpg"):
            pass
        elif os.path.isdir(path_to_folder + image) == True:
            pass
        else:
            convertVideoToImage(path_to_folder, path_to_folder)
            os.remove(path_to_folder + image)

def check_if_testable(path_to_folder):
    chopUpDataset(path_to_folder, path_to_folder, x=416, y=416, annotations=False)
    for image in os.listdir(path_to_folder):
        if image.endswith(".jpg"):
            width, height, _ = cv2.imread(path_to_folder + image).shape
            if width > 416 or height > 416:
                os.remove(path_to_folder + image)
            else:
                pass
        else:
            pass

