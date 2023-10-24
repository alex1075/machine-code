#! /usr/bin/python3
import os
import time
from code.convert import *
from code.helper.utils import *
from code.helper.annotations import *
from code.helper.imageTools import *
from code.helper.utils import *
from code.helper.yolo import *
from code.helper.fancy import *
from code.helper.config import *
from code.helper.threading import *
from code.helper.augment import *
from code.helper.annotations import *

def end_program():
    a = input('Do you have something else to do? (y/n)')
    if a == 'y':
        main()
    elif a == 'n':
        print('Exiting')
        banner_goodbye()
        clear()
        exit()

def prepare_all_training():
        clear()
        data_processing_banner()
        print('Preparing a dataset for training')
        question = input('Are you importing data from Roboflow? (y/n)')
        if question == 'y':
            ul = input('Past the dataset url here: ')
            path = input('Enter the path to the output folder: ')
            path = check_full_path(path)
            prepare_training(ul, path)
        if question == 'n':
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

def main(docker=False):
    if docker == True:
        print('Running containerised')
        path = '/media'
    # display_banner()
    # selection_program()
    clear()
    question = [inquirer.List('selection',
                           message=" Main machine interface, what do you wish to do?",
                           choices=['Convert multimedia data to JPEG images', 'Prepare a dataset for training', 'Dataset augmentation', 
                                    'Train a model', 'Test a model', 'Infer a model on biological data', 'Copy data over', 
                                    'Beta test a function', 'Exit'],
                       ),]
    a = inquirer.prompt(question)
    if a['selection'] == 'Convert multimedia data to JPEG images':
        clear()
        data_processing_banner()
        print('Converting multimedia data to JPEG images')
        if docker == False:
            path = input('Enter the path to the data: (remember to end with a /)')
            path = check_full_path(path)
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
                else:
                    print('File type not supported')
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
                            chopUpDataset(temp, out, 416, 416, False)
                            end_program()
                    elif cut == 'n':    
                        convertVideoToImage(path, out)
                        print('Video converted')
                        end_program()
                elif file.endswith('.jpg') or file.endswith('.png') or file.endswith('.tiff') or file.endswith('.bmp'):
                    if cut == 'y':
                        convert(path, True, 'temp/')
                        print('Images converted')
                        if ann == 'y':
                            chopUpDataset('temp/', out, 416, 416, True)
                            os.remove('temp/*')
                            end_program()
                        elif ann == 'n':
                            chopUpDataset('temp/', out, 416, 416, False)
                            os.remove('temp/*')
                            end_program()
                    elif cut == 'n':   
                        convert(path, True, out)
                        print('Images converted')
                        end_program()
                else:
                    print('File type not supported')
    elif a['selection'] == 'Prepare a dataset for training':
        prepare_all_training()
    elif a['selection'] == 'Train a model':
        clear()
        train_banner()
        prep = yes_no_question('Do you wish to prepare the folder first?')
        if prep == 'y':
            prepare_all_training()
        print('This is the folder with the Train/Test/Valid folders')
        print('and the obj.data, obj.names, and yolov4_10.cfg files')
        if docker == False:
            path = choose_folder('/home/as-hunt/Etra-Space/')
        path = check_full_path(path)
        w_choice = yes_no_question('Do you want to use the default weights to begin training?')
        if w_choice == 'y':
            if docker == True:
                weights = '/root/yolov4.conv.137'
            else:
                weights = '/home/as-hunt/Etra-Space/cfg/yolov4.conv.137'
        else:
            weights = input('Enter the path to the weights: ')
        weights = check_full_path(weights)
        print()
        a_choice = yes_no_question('Do you want to use the default arguments? -mjpeg_port 8090 -clear -dont_show')
        if a_choice == 'y':
            argus = ' -mjpeg_port 8090 -clear -dont_show'
        else:
            argus = input('Enter the arguments: ')
        epochs = choose_epochs()
        g_choice = yes_no_question('Do you want to generate training graph reports?')
        train_fancy(path, epochs, weights, argus)
        if g_choice == 'y':
            make_training_graphs(path + 'output.csv', path)
        train_complete_banner()    
    elif a['selection'] == 'Infer a model on biological data':
        clear()
        infer_banner()
        q2 = inquirer.prompt([inquirer.List('selection',
                           message="What do you wish to do?",
                           choices=['Analyze data over the network', 'Analyze data locally'],
                       ),])
        aa = q2['selection']
        if aa == 'Analyze data locally':
            if docker == False:
                path = input('Enter the path to the data: (remember to end with a /)')
                model = choose_folder('/home/as-hunt/Etra-Space/')
            else:
                path = '/media/'
                model = choose_folder('/media/')    
            name = choose_weights(model)
            a = yes_no_question('Do you want to save generated labels?')
            if a == 'y':
                save = True
            else:
                save = False
            clear()
            check_for_img(path)
            check_if_testable(path)    
            get_info(path, model, name, save)
        elif aa == 'Analyze data over the network':
            if docker == False:
                string = input('Enter the name of the file to search for: ')
                path = input('Enter the path where to copy the data: (remember to end with a /)')    
                get_file_over(path, string)
                clear()
                model = choose_folder('/home/as-hunt/Etra-Space/')
                name = choose_weights(model)
                a = yes_no_question('Do you want to save generated labels?')
                if a == 'y':
                    save = True
                else:
                    save = False
                clear()    
                check_for_img(path)
                check_if_testable(path)   
                get_info(path, model, name, save)
            elif docker == True:
                clear()
                error_banner()
                print(bcolors.ERROR + 'ERROR: not supported within a docker container')
                reset_color()    
        else:
            clear()
            error_banner()
            print(bcolors.ERROR + 'ERROR: Analyzing data over the network is not yet supported')
            reset_color()
    elif a['selection'] == 'Exit':
        end_program()
    elif a['selection'] == 'Copy data over':
        if docker == False:
            clear()
            string = input('Enter the name of the file to search for: ')
            path = input('Enter the path where to copy the data: (remember to end with a /)')    
            get_file_over(path, string)
            clear()
            end_program()
        elif docker == True:
            clear()
            error_banner()
            print(bcolors.ERROR + 'ERROR: not supported within a docker container')
            reset_color()
    elif a['selection'] == 'Beta test a function':
        clear()
        beta_banner()
        augmentations = choose_augmentations()
        path = '/home/as-hunt/machine-code/test1/'
        iterate_augment(path, augmentations, True)
    elif a['selection'] == 'Test a model':
        if docker == False:
            clear()
            test_banner()
            output_name = input('Enter the name of the output files: ')
            path = choose_folder('/home/as-hunt/Etra-Space/')
            test_fancy(path + '/', output_name)
        elif docker == True:
            clear()
            test_banner()
            output_name = input('Enter the name of the output files: ')
            path = choose_folder('/media/')
            test_fancy(path + '/', output_name)
    else:
        print('Invalid selection')
        time.sleep(2)
        main()

if __name__ == '__main__':
    main()