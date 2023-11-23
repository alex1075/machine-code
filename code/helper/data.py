import os
import shutil
import random
from code.convert import *
from code.helper.utils import *
from code.helper.fancy import *

def data_conversion(docker=False):
        print('Converting multimedia data to JPEG images')
        if docker == False:
            path = input('Enter the path to the data: (remember to end with a /)')
            path = check_full_path(path)
        elif docker == True:
            path = choose_folder('/media/')   
        temp = yes_no_question('Use temp folder?')
        for file in os.listdir(path):
            if temp == 'y':
                if file.endswith('.mp4'):
                    clear()
                    print('Converting video to image series')
                    temp = check_full_path('temp/')
                    convertVideoToImage(path, temp)
                    print('Video converted')
                    time.sleep(2)
                    clear()
                    cut = yes_no_question('Do you wish to cut the images into 416x416 pixels?')
                    ann = yes_no_question('Is the data annotated?')
                    if cut == 'y':
                        if ann == 'y':
                            os.mkdir('temp2/')
                            chopUpDataset(temp, 'temp2/', 416, 416, True)
                            os.system('rm -r ' + temp)
                            os.system('mv temp2/ ' + temp)
                            end_program()
                        elif ann == 'n':
                            os.mkdir('temp2/')
                            chopUpDataset(temp, 'temp2/', 416, 416, False)
                            os.system('rm -r ' + temp)
                            os.system('mv temp2/ ' + temp)
                            end_program()
                    elif cut == 'n':
                        end_program()
                elif file.endswith('.png') or file.endswith('.tiff') or file.endswith('.bmp') or file.endswith('.jpg'):
                    print('Converting images to JPG')
                    try:
                        temp = check_full_path('temp/')
                    except:
                        os.mkdir('temp/')
                        temp = check_full_path('temp/')    
                    convert(path, True, temp)
                    print('Images converted')
                    cut = yes_no_question('Do you wish to cut the images into 416x416 pixels?')
                    if cut == 'y':
                        ann = yes_no_question('Is the data annotated?')
                        if ann == 'y':
                            temp2 = 'temp2/'
                            try:
                                os.mkdir(temp2)
                            except:
                                pass
                            temp2 = check_full_path(temp2)
                            try:
                                chopUpDataset(temp, temp2, 416, 416, True)
                            except:
                                chopUpDataset(path, temp2, 416, 416, True)
                            os.system('rm -r ' + temp)
                            os.system('mv ' +temp2 + ' ' + temp)
                            end_program()
                        elif ann == 'n':
                            os.mkdir('temp2/')
                            chopUpDataset(temp, 'temp2/', 416, 416, False)
                            os.system('rm -r ' + temp)
                            end_program()
                    elif cut == 'n':
                        end_program()
            elif temp == 'n':
                if docker == False:
                    out = input('Enter the path to the output folder: (end with a /)')
                    out = check_full_path(out)
                else:
                    out = '/media/out/'
                    if os.path.isdir(out) == False:
                        os.system('mkdir ' + out)
                    else:
                        pass    
                    out = check_full_path(out)
                cut = yes_no_question('Do you wish to cut the images into 416x416 pixels?')
                ann = yes_no_question('Is the data annotated?')
                if file.endswith('.mp4'):
                    print('Converting video to image series')
                    print('Save path: ' + out)
                    if cut == 'y':
                        temp = check_full_path(out)
                        convertVideoToImage(path, 'temp/')
                        print('Video converted')
                        if ann == 'y':
                            chopUpDataset('temp/', out, 416, 416, True)
                            end_program()
                        elif ann == 'n':
                            chopUpDataset('temp/', out, 416, 416, False)
                            end_program()
                    elif cut == 'n':    
                        convertVideoToImage(path, out)
                        print('Video converted')
                        end_program()
                elif file.endswith('.jpg') or file.endswith('.png') or file.endswith('.tiff') or file.endswith('.bmp'):
                    if cut == 'y':
                        convert(path, True, path)
                        print('Images converted')
                        if ann == 'y':
                            chopUpDataset(path, out, 416, 416, True)
                            # os.remove('temp/*')
                            end_program()
                        elif ann == 'n':
                            chopUpDataset(path, out, 416, 416, False)
                            os.remove('temp/*')
                            end_program()
                    elif cut == 'n':   
                        convert(path, True, out)
                        print('Images converted')
                        end_program()

def prepare_all_training(docker=False):
        clear()
        data_processing_banner()
        print('Preparing a dataset for training')
        if docker == False:
            path = input('Enter the path to the output folder: ')
            path = check_full_path(path)
        elif docker == True:
            print('Running containerised. Using /media/ as the path, where would you save on /media/?') 
            path = input('Enter the path to the output folder: ')
            path = check_full_path(path)    
        question = yes_no_question('Are you importing data from Roboflow?')
        if question == 'y':
            ul = input('Past the dataset url here: ')
            prepare_training(ul, path)
        if question == 'n':
            if docker == False:
                print('Make sure the images are annotated and within the same folder')
                path = input('Enter the path to the data: (remember to end with a /)')
                path = check_full_path(path)
                temp = input('Use temp folder? (y/n)')
                if temp == 'y':
                    temp = 'temp/'
                    temp = check_full_path(temp)
                    save_dir = os.getcwd() + '/' + temp
                    save_dir = check_full_path(save_dir)
                    import_names(path, True, temp)
                    select_yolo_version(temp + 'obj.names', temp, 100)
                    make_obj_data(path, True, save_dir)
                else:
                    out = input('Enter the path to the output folder: ')
                    out = check_full_path(out)
                    import_names(path, True, out)
                    select_yolo_version(out + 'obj.names', out, 100)
                    make_obj_data(path, True, out)
            elif docker == True:
                print('Run program locally to prepare data locally')        
 
def split_images_and_annotations(source_folder, destination_folders):
    '''Splits images and annotations from one main folder to multiple destination folders
    Args:
        source_folder: the folder containing the images and annotations
        destination_folders: a list of folders to copy the images and annotations to

    Returns:
        None
    '''
    # Create destination folders if they don't exist
    for folder in destination_folders:
        os.makedirs(folder, exist_ok=True)
    # Get a list of all image files in the source folder
    image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]
    for file in os.listdir(source_folder):
        if file == 'classes.txt': 
            classes = True
            break
        elif file == '_darknet.labels':
            shutil.copy(source_folder + '/' + file, source_folder + '/classes.txt')
            classes = True
            break
        else:
            classes = False
    print('Classes: ', classes)               
    # Shuffle the image files randomly
    random.shuffle(image_files)
    # Calculate the number of images in each split
    split_size = len(image_files) // len(destination_folders)
    # Copy images and annotations to the destination folders
    for i, folder in enumerate(destination_folders):
        start_index = i * split_size
        end_index = (i + 1) * split_size if i < len(destination_folders) - 1 else len(image_files)
        for j in range(start_index, end_index):
            image_file = image_files[j]
            annotation_file = image_file.replace('.jpg', '.txt')
            source_image_path = os.path.join(source_folder, image_file)
            source_annotation_path = os.path.join(source_folder, annotation_file)
            destination_image_path = os.path.join(folder, image_file)
            destination_annotation_path = os.path.join(folder, annotation_file)
            shutil.copy(source_image_path, destination_image_path)
            shutil.copy(source_annotation_path, destination_annotation_path)
            if classes == True:
                shutil.copy(source_folder + '/classes.txt', folder + '/classes.txt')

def combine_folders(source_folders, destination_folder):
    ''' Combines multiple folders into one
    Args:
        source_folders: a list of folders to copy the images and annotations from
        destination_folder: the folder to copy the images and annotations to

    Returns:
        None    
    '''
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    # Iterate over each source folder
    for source_folder in source_folders:
        # Get a list of all image files in the source folder
        image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]
        for file in os.listdir(source_folder):
             if file == 'classes.txt': 
                 shutil.copy(source_folder + '/' + file, destination_folder + '/classes.txt')
             elif file == '_darknet.labels':
                shutil.copy(source_folder + '/' + file, destination_folder + '/classes.txt')
             else:
                 pass
        # Copy images and annotations to the destination folder
        for image_file in image_files:
            annotation_file = image_file.replace('.jpg', '.txt')
            source_image_path = os.path.join(source_folder, image_file)
            source_annotation_path = os.path.join(source_folder, annotation_file)
            destination_image_path = os.path.join(destination_folder, image_file)
            destination_annotation_path = os.path.join(destination_folder, annotation_file)
            shutil.copy(source_image_path, destination_image_path)
            shutil.copy(source_annotation_path, destination_annotation_path)

def split_to_X_folders(source_folder, destination_folder, number=int):
    '''Split a folder into X folders
    Args:
        source_folder: the folder to split
        destination_folder: the folder to copy the images and annotations to
        number: the number of folders to split into

    Returns:
        None
    '''
    destination_folders = [f"{destination_folder}/f{i+1}" for i in range(number)]
    split_images_and_annotations(source_folder, destination_folders)            

def combine_three_folders(source_folders, destination_folder, n1=int, n2=int, n3=int, n4=int, n5=int, n6=int, n7=int, n8=int, n9=int, n10=int):
    '''Combine X folders into one
    Args:
        source_folders: a list of folders to copy the images and annotations from
        destination_folder: the folder to copy the images and annotations to
        number: the number of folders to combine

    Returns:
        None
    '''
    source_folders = [f"{source_folders}/f{n1}", f"{source_folders}/f{n2}", f"{source_folders}/f{n3}"]
    combine_folders(source_folders, destination_folder)    