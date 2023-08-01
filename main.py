import os
from code.convert import *
from code.helper.utils import *
from code.helper.annotations import *
from code.helper.imageTools import *
from code.helper.utils import *
from code.helper.yolo import *

def main():
    print('Main machine interface, what do you wish to do?')
    print('1. Convert multimedia data to JPEG images')
    print('2. Prepare a dataset for training (image cropping, annotation file modification, etc)')
    print('3. Train a model')
    print('4. Test a model')
    print('5. Infer a model on biological data')
    print('6. Exit')
    a = input('Enter a a selection: ')
    if a == '1':
        print('Converting multimedia data to JPEG images')
        print('Enter the path to the data: ')
        print('Remember to use quotation marks and end the path with a /')
        path = input()
        print('Use temp folder? (y/n)')
        temp = input()
        for file in os.listdir(path):
            if temp == 'y':
                if file.endswith('.mp4'):
                    convertVideoToImage(path, 'temp/')
                elif file.endswith('.png') or file.endswith('.tiff') or file.endswith('.bmp') or file.endswith('.jpg'):
                    convert(path)
                else:
                    print('File type not supported')
            else:
                print('Enter the path to the output folder: ')
                print('Remember to use quotation marks and end the path with a /')
                out = input()
                if file.endswith('.mp4'):
                    convertVideoToImage(path, out)
                elif file.endswith('.jpg') or file.endswith('.png') or file.endswith('.tiff') or file.endswith('.bmp'):
                    convert(path, out)
                else:
                    print('File type not supported')
    elif a == '2':
        print('Preparing a dataset for training')
        print('Make sure the images are annotated and within the same folder')
        print('Enter the path to the images: ')
        print('Remember to use quotation marks and end the path with a /')
        path = input()
        print('Use temp folder? (y/n)')
        temp = input()
        if temp == 'y':
            chopUpDataset(path, tempFolder='temp/')
        else:
            print('Enter the path to the output folder: ')
            print('Remember to use quotation marks and end the path with a /')
            out = input()
            chopUpDataset(path, outFolder=out)
    elif a == '3':
        print('Training a model')
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
        print('Testing a model')
        print('Function not yet implemented')
    elif a == '5':
        print('Infering a model on biological data')
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
            print('Enter the path to the output folder: ')
            print('Remember to use quotation marks and end the path with a /')
            out = input()
            print('Do you want to save generated labels? (y/n)')
            a = input()
            if a == 'y':
                save = True
            else:
                save = False
            get_info(path, model, name, out, save)
        else:
            print('Analyzing data over the network is not yet supported')
    elif a == '6':
        print('Exiting')
        exit()
    else:
        print('Invalid selection')
        main()



if __name__ == '__main__':
    main()


