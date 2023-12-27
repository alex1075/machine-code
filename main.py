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
import code.data.config as config
from code.helper.data import *

def main(docker=False):
    docker = config.containerised
    clear()
    if docker == True:
        print('Running containerised.')
        path = '/media'
    if docker == False:
        choice = ['Convert multimedia data to JPEG images', 'Prepare a dataset for training', 'Dataset augmentation', 
                                    'Train a model', 'Test a model', 'Get training info', 'Infer a model on biological data', 'Copy data over', 
                                    'Train with five fold validation', 'Test 5 fold validation', 'Beta test a function', 'Exit']
    elif docker == True:
        choice = ['Convert multimedia data to JPEG images', 'Prepare a dataset for training', 'Dataset augmentation', 
                                    'Train a model', 'Test a model', 'Get training info', 'Infer a model on biological data',
                                    'Train with five fold validation', 'Test 5 fold validation', 'Exit']
    question = [inquirer.List('selection',
                           message="Main machine interface, what do you wish to do?",
                           choices=choice,
                       ),]
    a = inquirer.prompt(question)
    if a['selection'] == 'Convert multimedia data to JPEG images':
        clear()
        data_processing_banner()
        data_conversion()
    elif a['selection'] == 'Prepare a dataset for training':
        prepare_all_training(docker)
    elif a['selection'] == 'Train a model':
        clear()
        train_banner()
        prep = yes_no_question('Do you wish to prepare the folder first?')
        if prep == 'y':
            prepare_all_training(docker)
        print('This is the folder with the Train/Test/Valid folders')
        print('and the obj.data, obj.names, and yolov4_10.cfg files')
        if docker == False:
            path = choose_folder('/home/as-hunt/Etra-Space/')
        elif docker == True:
            path = choose_folder('/media/')
        path = check_full_path(path)
        w_choice = yes_no_question('Do you want to use the default weights to begin training?')
        if w_choice == 'y':
            if docker == True:
                weights = '/root/yolov4.conv.137'
            else:
                weights = '/home/as-hunt/Etra-Space/cfg/yolov4.conv.137'
        else:
            if docker == False:
                weights = input('Enter the path to the weights: ')
            elif docker == True:
                weights = input('Remember to have the files mounted in /media. Enter the path to the weights: ')
        weights = check_full_path(weights)
        a_choice = yes_no_question('Do you want to use the default arguments? -mjpeg_port 8090 -clear -dont_show')
        if a_choice == 'y':
            argus = ' -mjpeg_port 8090 -clear -dont_show'
        else:
            argus = input('Enter the arguments: ')
        epochs = choose_epochs()
        g_choice = yes_no_question('Do you want to generate training graph reports?')
        if g_choice == 'y':
            train_fancy(path, epochs, weights, argus, True)
            make_training_graphs(path + 'output.csv', path)
        elif g_choice == 'n':
            train_fancy(path, epochs, weights, argus, False)
        train_complete_banner()    
    elif a['selection'] == 'Infer a model on biological data':
        clear()
        infer_banner()
        if docker == False:
            q2 = inquirer.prompt([inquirer.List('selection',
                           message="What do you wish to do?",
                           choices=['Analyze data over the network', 'Analyze data locally'],
                       ),])
            aa = q2['selection']
        elif docker == True:
            aa = 'Analyze data locally'

        if aa == 'Analyze data locally':
            if docker == False:
                path = input('Enter the path to the data: (remember to end with a /)')
                model = choose_folder('/home/as-hunt/Etra-Space/')
            elif docker == True:
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
    elif a['selection'] == 'Exit':
        end_program()
    elif a['selection'] == 'Copy data over':
            clear()
            string = input('Enter the name of the file to search for: ')
            path = input('Enter the path where to copy the data: (remember to end with a /)')    
            get_file_over(path, string)
            clear()
            end_program()
    elif a['selection'] == 'Beta test a function':
        clear()
        beta_banner()
        qq = yes_no_question('Is the origin folder in Etra-Space?')
        if qq == 'y':
            path = '/home/as-hunt/Etra-Space/'
            path1 = choose_folder(path)
        else:
            path = '/home/as-hunt/'
            path1 = choose_folder(path)           
        output_name = input('Enter the name of the output files: ')    
        path = check_full_path(path1)
        epochs = choose_epochs()
        multiprocess_test_5_fold_valdiation_cv2(path, output_name, epochs)
    elif a['selection'] == 'Test a model':
        clear()
        test_banner()
        cv2q = yes_no_question('Do you want to use OpenCV for the test?')
        output_name = input('Enter the name of the output files: ')
        if docker == False:
            path = choose_folder('/home/as-hunt/Etra-Space/')
        elif docker == True:    
            path = choose_folder('/media/')
        if cv2q == 'y':
            cv2_test_fancy(path + '/', output_name)    
        else:
            test_fancy(path + '/', output_name)
    elif a['selection'] == 'Dataset augmentation':
        clear()
        qq = yes_no_question('Is the origin folder in Etra-Space?')
        if qq == 'y':
            path = '/home/as-hunt/Etra-Space/'
            path1 = choose_folder(path)
        else:
            path = '/home/as-hunt/'
            path1 = choose_folder(path)
        q1 = yes_no_question('Is the data annotated?')
        if q1 == 'y':
            annot = True
        else:
            annot = False
        augments = choose_augmentations()
        iterate_augment(path1, augments, annot)    
    elif a['selection'] == 'Train with five fold validation':
        clear()
        qq = yes_no_question('Is the origin folder in Etra-Space?')
        if qq == 'y':
            path = '/home/as-hunt/Etra-Space/'
            path1 = choose_folder(path)
        else:
            path = '/home/as-hunt/'
            path1 = choose_folder(path)
        path2 = input('Enter the path to the working folder: ')
        path2 = check_full_path(path2)
        os.makedirs(path2, exist_ok=True)
        w_choice = yes_no_question('Do you want to use the default weights to begin training?')
        if w_choice == 'y':
                weights = '/home/as-hunt/Etra-Space/cfg/yolov4.conv.137'
        else:
                weights = input('Remember to have the files mounted in /media. Enter the path to the weights: ')
        weights = check_full_path(weights)
        a_choice = yes_no_question('Do you want to use the default arguments? -mjpeg_port 8090 -dont_show')
        if a_choice == 'y':
            argus = ' -mjpeg_port 8090 -dont_show'
        else:
            argus = input('Enter the arguments: ')
        epochs = choose_epochs()
        g_choice = yes_no_question('Do you want to generate training graph reports?') 
        if g_choice == 'y':
            train_5_fold_validation(path1, path2, epochs, weights, argus, True)
            for i in range(1,6,1):
                make_training_graphs(path2 + f'/{i}/' + 'output.csv', path2)
        elif g_choice == 'n':
            train_5_fold_validation(path1, path2, epochs, weights, argus, False)   
    elif a['selection'] == 'Test 5 fold validation':
        clear()
        cv2q = yes_no_question('Do you want to use OpenCV for the test?')
        qq = yes_no_question('Is the origin folder in Etra-Space?')
        if qq == 'y':
            path = '/home/as-hunt/Etra-Space/'
            path1 = choose_folder(path)
        else:
            path = '/home/as-hunt/'
            path1 = choose_folder(path)           
        output_name = input('Enter the name of the output files: ')    
        path = check_full_path(path1)
        epochs = choose_epochs()
        if cv2q == 'y':
            test_5_fold_validation_cv2(path, output_name, epochs)
        elif cv2q == 'n':
            test_5_fold_validation(path, output_name, epochs)        
    elif a['selection'] == 'Get training info':
        clear()
        path = input('Enter the path to the data: (remember to end with a /)')
        output_name = input('Enter the name of the output files: ')    
        path = check_full_path(path)
        test_training_epochs(path, output_name)                    
    else:
        print('Invalid selection')
        time.sleep(2)
        main()

if __name__ == '__main__':
    main()
