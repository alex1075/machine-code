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
            end_program()
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
                prepare_cfg('code/data/yolov4.cfg', temp + 'obj.names', temp, 100, 'yolov4_10.cfg')
                make_obj_data(path, True, save_dir)
                end_program()
            else:
                out = input('Enter the path to the output folder: ')
                out = check_full_path(out)
                import_names(path, True, out)
                prepare_cfg('code/data/yolov4.cfg', out + 'obj.names', out, 100, 'yolov4_10.cfg')
                make_obj_data(path, True, out)
                end_program()

def main():
    display_banner()
    selection_program()
    print('Main machine interface, what do you wish to do?')
    print('1. Convert multimedia data to JPEG images')
    print('2. Prepare a dataset for training (image cropping, annotation file modification, etc)')
    print('3. Train a model')
    print('4. Test a model')
    print('5. Infer a model on biological data')
    print('6. Exit')
    print('7. beta test a function')
    a = input('Enter a a selection: ')
    if a == '1':
        clear()
        data_processing_banner()
        print('Converting multimedia data to JPEG images')
        path = input('Enter the path to the data: (remember to end with a /)')
        path = check_full_path(path)
        temp = input('Use temp folder? (y/n)')
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
                    cut = input('Do you wish to cut the images into 416x416 pixels? (y/n)')
                    ann = input('Is the data annotated? (y/n)')
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
                    temp = check_full_path('temp/')
                    convert(path, True, temp)
                    print('Images converted')
                    print('Do you wish to cut the images into 416x416 pixels? (y/n)')
                    cut = input()
                    if cut == 'y':
                        ann = input('Is the data annotated? (y/n)')
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
                            end_program()
                    elif cut == 'n':
                        end_program()
                else:
                    print('File type not supported')
            elif temp == 'n':
                out = input('Enter the path to the output folder: (end with a /)')
                out = check_full_path(out)
                cut = input('Do you wish to cut the images into 416x416 pixels? (y/n)')
                ann = input('Is the data annotated? (y/n)')
                if file.endswith('.mp4'):
                    print('Converting video to image series')
                    print('Save path: ' + out)
                    if cut == 'y':
                        temp = os.mkdir('tmp/')
                        temp = check_full_path(temp)
                        convertVideoToImage(path, 'temp/')
                        print('Video converted')
                        if ann == 'y':
                            chopUpDataset('temp/', out, 416, 416, True)
                            os.system('rm -r ' + temp)
                            end_program()
                        elif ann == 'n':
                            chopUpDataset(temp, out, 416, 416, False)
                            os.system('rm -r ' + temp)
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
    elif a == '2':
        prepare_all_training()
    elif a == '3':
        clear()
        train_banner()
        prep = input('Do you wish to prepare the folder first? (y/n)')
        if prep == 'y':
            prepare_all_training()
        print('This is the folder with the Train/Test/Valid folders')
        print('and the obj.data, obj.names, and yolov4_10.cfg files')
        path = input('Enter the path to the data: (remember to end with a /)')
        path = check_full_path(path)
        w_choice = input('Do you want to use the default weights to begin training? (y/n)')
        if w_choice == 'y':
            weights = '/home/as-hunt/Etra-Space/cfg/yolov4.conv.137'
        else:
            weights = input('Enter the path to the weights: ')
        weights = check_full_path(weights)
        print()
        a_choice = input('Do you want to use the default arguments? -mjpeg_port 8090 -clear -dont_show (y/n)')
        if a_choice == 'y':
            argus = ' -mjpeg_port 8090 -clear -dont_show'
        else:
            argus = input('Enter the arguments: ')
        e_choice = input('Do you want to use the default number of epochs? (y/n)')
        if e_choice == 'y':
            epochs = 10000
        else:
            epochs = input('Enter the number of epochs: ')
            epochs = int(epochs)
        g_choice = input('Do you want to generate training graph reports? (y/n)')
        train_fancy(path, epochs, weights, argus)
        if g_choice == 'y':
            make_training_graphs(path + 'output.csv', path)
        train_complete_banner()    
    elif a == '4':
        clear()
        test_banner()
        print('Testing a model')
        error_banner()
        print(bcolors.ERROR + 'Function not yet implemented')
        reset_color()
    elif a == '5':
        clear()
        infer_banner()
        # print('Infering a model on biological data')
        print('Has the data been copied to the local drive? (y/n)')
        copy = input()
        if copy == 'y':
            print('Enter the path to the data: ')
            print('Remember to use quotation marks and end the path with a /')
            path = input('Enter the path to the data: (remember to end with a /)')
            print('Enter the path to the model to be used: ')
            print('Remember to use quotation marks')
            model = input()
            print('Enter the name of the model to use: ')
            print('Remember to use quotation marks')
            name = input()
            print('Do you want to save generated labels? (y/n)')
            a = input()
            if a == 'y':
                save = True
            else:
                save = False
            get_info(path, model, name, save)
        else:
            clear()
            error_banner()
            print(bcolors.ERROR + 'ERROR: Analyzing data over the network is not yet supported')
            reset_color()
    elif a == '6':
        end_program()
    elif a == '7':
        clear()
        warnings_banner()
        print(bcolors.WARNING + 'Beta testing a function, it may not work')
        reset_color()
        try:
            checkAllImg('temp/', 416, 416)
        except Exception as e:
            print(bcolors.ERROR + 'Something went wrong')
            a = input('Do you want to see the error? (y/n)')
            if a == 'y':
                print(bcolors.ERROR + str(e))
    else:
        print('Invalid selection')
        time.sleep(2)
        main()


if __name__ == '__main__':
    main()


