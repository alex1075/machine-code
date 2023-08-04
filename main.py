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
        print('Enter the path to the data: ')
        print('Remember to use quotation marks and end the path with a /')
        path = input()
        print('Use temp folder? (y/n)')
        temp = input()
        for file in os.listdir(path):
            if temp == 'y':
                if file.endswith('.mp4'):
                    clear()
                    print('Converting video to image series')
                    print('Save path: ' + os.getcwd() + '/temp/')
                    convertVideoToImage(path, 'temp/')
                    print('Video converted')
                    clear()
                    cut = input('Do you wish to cut the images into 416x416 pixels? (y/n)')
                    ann = input('Is the data annotated? (y/n)')
                    if cut == 'y':
                        if ann == 'y':
                            chopUpDataset('temp/', 'temp2/', 416, 416, True)
                            end_program()
                        elif ann == 'n':
                            chopUpDataset('temp/', 'temp2/', 416, 416, False)
                            end_program()
                    elif cut == 'n':
                        end_program()
                elif file.endswith('.png') or file.endswith('.tiff') or file.endswith('.bmp') or file.endswith('.jpg'):
                    print('Converting images to JPG')
                    print('Save path: ' + os.getcwd() + '/temp/')
                    convertVideoToImage(path, 'temp/')
                    convert(path, True, 'temp/')
                    print('Images converted')
                    print('Do you wish to cut the images into 416x416 pixels? (y/n)')
                    cut = input()
                    if cut == 'y':
                        ann = input('Is the data annotated? (y/n)')
                        if ann == 'y':
                            chopUpDataset('temp/', 'temp/', 416, 416, True)
                            end_program()
                        elif ann == 'n':
                            chopUpDataset('temp/', 'temp/', 416, 416, False)
                            end_program()
                    elif cut == 'n':
                        end_program()
                else:
                    print('File type not supported')
            elif temp == 'n':
                print('Enter the path to the output folder: ')
                print('Remember to use quotation marks and end the path with a /')
                out = input()
                cut = input('Do you wish to cut the images into 416x416 pixels? (y/n)')
                ann = input('Is the data annotated? (y/n)')
                if file.endswith('.mp4'):
                    print('Converting video to image series')
                    print('Save path: ' + out)
                    if cut == 'y':
                        convertVideoToImage(path, 'temp/')
                        print('Video converted')
                        if ann == 'y':
                            chopUpDataset('temp/', out, 416, 416, True)
                            os.remove('temp/*')
                            end_program()
                        elif ann == 'n':
                            chopUpDataset('temp/', out, 416, 416, False)
                            os.remove('temp/*')
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
        clear()
        data_processing_banner()
        print('Preparing a dataset for training')
        print('Make sure the images are annotated and within the same folder')
        print('Enter the path to the images: ')
        print('Remember to use quotation marks and end the path with a /')
        path = input()
        print('Use temp folder? (y/n)')
        temp = input()
        if temp == 'y':
            temp = 'temp/'
            save_dir = os.getcwd() + '/' + temp
            import_names(path, True, temp)
            prepare_cfg('code/data/yolov4.cfg', temp + 'obj.names', temp, 100, 'yolov4_10.cfg')
            make_obj_data(path, True, save_dir)
        else:
            print('Enter the path to the output folder: ')
            print('Remember to use quotation marks and end the path with a /')
            out = input()
            import_names(path, True, out)
            prepare_cfg('code/data/yolov4.cfg', out + 'obj.names', out, 100, 'yolov4_10.cfg')
            make_obj_data(path, True, out)
    elif a == '3':
        clear()
        train_banner()
        # print('Training a model')
        print('Do you wish to prepare the folder first? (y/n)')
        prep = input()
        if prep == 'y':
            print('Function not yet implemented')
        print('Enter the path to the training folder: ')
        print('This is the folder with the Train/Test/Valid folders')
        print('and the obj.data, obj.names, and yolov4_10.cfg files')
        print('Remember to use quotation marks and end the path with a /')
        path = input()
        print('Do you want to use the default weights to begin training? (y/n)')
        w_choice = input()
        if w_choice == 'y':
            weights = '/home/as-hunt/Etra-Space/cfg/yolov4.conv.137'
        else:
            print('Enter the path to the weights: ')
            print('Remember to use quotation marks')
            weights = input()
        print('Do you want to use the default arguments? (y/n)')
        a_choice = input()
        if a_choice == 'y':
            argus = ' -mjpeg_port 8090 -clear -dont_show'
        else:
            print('Enter the arguments: ')
            print('Remember to use quotation marks')
            argus = input()
        print('Do you want to use the default number of epochs? (y/n)')
        e_choice = input()
        if e_choice == 'y':
            epochs = 10000
        else:
            print('Enter the number of epochs: ')
            epochs = input()
        print('Do you want to generate training graph reports? (y/n)')
        g_choice = input()
        train_fancy(path, epochs, weights, argus)
        if g_choice == 'y':
            make_training_graphs(path + 'output.csv', path)
    elif a == '4':
        clear()
        test_banner()
        print('Testing a model')
        error_banner()
        print(bcolors.ERROR + 'Function not yet implemented')
    elif a == '5':
        clear()
        infer_banner()
        # print('Infering a model on biological data')
        print('Has the data been copied to the local drive? (y/n)')
        copy = input()
        if copy == 'y':
            print('Enter the path to the data: ')
            print('Remember to use quotation marks and end the path with a /')
            path = input()
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
            print('ERROR: Analyzing data over the network is not yet supported')
    elif a == '6':
        print('Exiting')
        exit()
    elif a == '7':
        clear()
        warnings_banner()
        print(bcolors.WARNING + 'Beta testing a function, it may not work')
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


