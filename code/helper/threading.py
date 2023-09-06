import os
import time 
import shutil
import threading
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
from code.convert import *
from code.helper.utils import *
from code.helper.annotations import *
from code.helper.reports import *
from code.helper.config import *
from code.helper.yolo import *
from code.helper.watch import *
from code.helper.imageTools import *

def multi_thread_crop(x, y, path, save_path, annotations=False):
    path = check_full_path(path)
    save_path = check_full_path(save_path)
    list_img=[img for img in os.listdir(path) if img.endswith('.jpg')==True]
    thread_list = []
    proc = os.cpu_count()
    if annotations == True:
        shutil.copy(path + "classes.txt", save_path)
    else:
        pass
    path = check_full_path(path)
    lengt = len(list_img)
    for i in range(proc):
        listy = list_img[int(i * lengt / proc):int((i + 1) * lengt / proc)]
        thread = threading.Thread(target=crop_image_list, args=(x, y, listy, save_path, annotations))
        thread_list.append(thread)
        thread_list[i].start()
        print('Started thread ' + str(i))

    for i in thread_list:
        thread.join()



def multi_file_Video_convert(path):
    tic = time.perf_counter()
    path = check_full_path(path)
    proc = os.cpu_count()
    print('Number of processors: ' + str(proc))
    videos = glob.glob(path + '*.mp4')
    count = len(videos)
    print(count)
    thread_list = []
    a = 0
    cpu = 0
    while a <= len(videos):
        for cpu in range(proc):
            try:
                thread = threading.Thread(target=convertAVideoToImage, args=(videos[a], path))
                thread_list.append(thread)
                thread_list[-1].start()
            except:
                for i in thread_list:
                    thread.join()
            cpu += 1
            a += 1
        for i in thread_list:
            thread.join()
            cpu -= (len(videos) - a)
            if cpu <= 0:
                cpu = 0
    print('All threads finished')
    toc = time.perf_counter()
    print(f"Finished in {toc - tic:0.4f} seconds")        
    

def list_check_if_testable(listy):
    for image in listy:
        if image.endswith(".jpg"):
            width, height, _ = cv2.imread(image).shape
            if width > 416 or height > 416:
                os.remove(image)
            else:
                pass
        else:
            pass

def multi_thread_check_if_testable(path_to_folder):
    path_to_folder = check_full_path(path_to_folder)
    multi_thread_crop(416, 416, path_to_folder, path_to_folder, annotations=False)
    list_img=[img for img in os.listdir(path_to_folder) if img.endswith('.jpg')==True]
    thread_list = []
    proc = os.cpu_count()
    lengt = len(list_img)
    for i in range(proc):
        listy = list_img[int(i * lengt / proc):int((i + 1) * lengt / proc)]
        thread = threading.Thread(target=list_check_if_testable, args=(listy, path_to_folder))
        thread_list.append(thread)
        thread_list[i].start()
        print('Started thread ' + str(i))

    for i in thread_list:
        thread.join()