import os 

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

def import_names(path_to_data):
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
                    make_obj_names(names, path_to_data)
        elif file == '_darknet.labels':
            with open(path_to_data + file) as f:
                for line in f:
                    names.append(line.strip())
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

