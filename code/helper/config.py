import os 

def prep(file="train.txt", path="/home/as-hunt/Etra-Space/white-thirds/train/"):
    '''Prepares a txt file for training

    args:
    file: the name of the file to be created
    path: the path to the folder containing the images
    '''
    filoo = open(path + file, 'w')
    for image in os.listdir(path):
        if image.endswith(".jpg"):
            filoo.write(path + image + "\n")
    filoo.close()

def change_line(file, line_number, text, save_as_new_file=False, new_file_name='yolo_temp.cfg'):
    '''Changes a specific line within a text file
    
    args:
        file: the file to be changed
        line_number: the line number to be changed
        text: the text to be inserted
        save_as_new_file: if True, the file will be saved as a new file
        new_file_name: the name of the new file
        '''
    lines = open(file, 'r').readlines()
    lines[line_number] = text
    if save_as_new_file == True:
        with open(new_file_name, 'w') as f:
            f.writelines(lines)
    else:
        with open(file, 'w') as f:
            f.writelines(lines)

def make_obj_names(names=['ECHY', 'ERY', 'LYM', 'MON', 'NEU', 'PLT'], path='temp/'):
        '''Creates a obj.names file from a list of names

        args:
        names: an array of names to be written to the file
        path: the path to the file
        '''
        names.sort()
        with open(path + 'obj.names', 'w') as f:
            for name in names:
                f.write(name + '\n')

def import_names(path_to_data, save_elsewhere=False, save_path='temp/'):
    '''Imports the names from a folder containing the data for training

    args:
    path_to_data: the path to the folder containing the data
    '''
    names = []
    os.listdir(path_to_data)
    for file in os.listdir(path_to_data):
        if file == 'classes.txt':
            with open(path_to_data + file) as f:
                for line in f:
                    names.append(line.strip())
                    if save_elsewhere == True:
                        make_obj_names(names, save_path)
                    else:
                        make_obj_names(names, path_to_data)
        elif file == '_darknet.labels':
            with open(path_to_data + file) as f:
                for line in f:
                    names.append(line.strip())
                    if save_elsewhere == True:
                        make_obj_names(names, save_path)
                    else:
                        make_obj_names(names, path_to_data)
        else:
            pass           

def prepare_cfg(template_cfg='code/data/yolov4.cfg', obj_names='temp/obj.names', output_folder='temp/', epochs_to_run_for=100, output_name='yolov4_10.cfg'):
    '''Prepares the cfg file for training
    
    args:
    template_cfg: the path to the template cfg file
    obj_names: the path to the obj.names file
    output_folder: the path to the output folder
    epochs_to_run_for: the number of epochs to run for
    '''
    with open(obj_names) as f:
        names = f.readlines()
        count = len(names) + 1
    filters = ((count + 5) * 3)    
    change_line(template_cfg, 19, 'max_batches = ' + str(epochs_to_run_for) + '\n', True, output_folder + output_name)
    change_line(output_folder + output_name, 21, 'steps = ' + str(round(epochs_to_run_for * 0.8)) + ',' + str(round(epochs_to_run_for * 0.9)) + '\n')
    change_line(output_folder + output_name, 962, 'filters = ' + str(filters) + '\n')
    change_line(output_folder + output_name, 969, 'classes = ' + str(count) + '\n')
    change_line(output_folder + output_name, 1051, 'filters = ' + str(filters) + '\n')
    change_line(output_folder + output_name, 1057, 'classes = ' + str(count) + '\n')
    change_line(output_folder + output_name, 1138, 'filters = ' + str(filters) + '\n')
    change_line(output_folder + output_name, 1145, 'classes = ' + str(count) + '\n')

def make_obj_data(path_to_data, save_elsewhere=False, save_path='temp/'):
    '''Creates the obj.data file for training
    will read the folder passed to find the number of classes
    the presence of the train, test, valid and backup folders

    If the backup folder does not exist, it will be created

    It will check for the presence of the train.txt, test.txt 
    and valid.txt files. If they do not exist, they will be created

    args:
    path_to_data: the path to the folder containing the data
    save_elsewhere: if True, the file will be saved elsewhere
    save_path: the path to save the file to
    '''
    os.chdir(path_to_data)
    if os.path.exists(path_to_data + 'obj.data'):
        count = len(open(path_to_data + 'obj.names').readlines + 1)
        names = os.getcwd(path_to_data + 'obj.names')
    if os.path.exists(path_to_data + 'train/'):
        if os.path.exists(path_to_data + 'train/train.txt'):
            train = os.getcwd(path_to_data + 'train/train.txt')
        else:
            prep('train.txt', path_to_data + 'train/')
            train = os.getcwd(path_to_data + 'train/train.txt')
    if os.path.exists(path_to_data + 'test/'):
        if os.path.exists(path_to_data + 'test/test.txt'):
            test = os.getcwd(path_to_data + 'test/test.txt')
        else:
            prep('test.txt', path_to_data + 'test/')
            test = os.getcwd(path_to_data + 'test/test.txt')
    if os.path.exists(path_to_data + 'valid/'):
        if os.path.exists(path_to_data + 'valid/valid.txt'):
            valid = os.getcwd(path_to_data + 'valid/valid.txt')
        else:
            prep('valid.txt', path_to_data + 'valid/')
            valid = os.getcwd(path_to_data + 'valid/valid.txt')
    if os.path.exists(path_to_data + 'backup/'):
        backup = os.getcwd(path_to_data + 'backup/')
    else:
        os.mkdir('backup')
        backup = os.getcwd(path_to_data + 'backup/')
    if save_elsewhere == True:
        with open(save_path + 'obj.data', 'w') as f:
            f.write('classes = ' + str(count) + '\n')
            f.write('train = ' + train + '\n')
            f.write('valid = ' + valid + '\n')
            f.write('# valid = ' + test + '\n')
            f.write('names = ' + names + '\n')
            f.write('backup = ' + backup + '\n')
    else:
        with open(path_to_data + 'obj.data', 'w') as f:
            f.write('classes = ' + str(count) + '\n')
            f.write('train = ' + train + '\n')
            f.write('valid = ' + valid + '\n')
            f.write('# valid = ' + test + '\n')
            f.write('names = ' + names + '\n')
            f.write('backup = ' + backup + '\n')