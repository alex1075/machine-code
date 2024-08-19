import os
import cv2
import tqdm
import subprocess
import shutil
import warnings
import threading
import pandas as pd
from code.helper.annotations import *
from code.helper.reports import *
from code.helper.utils import *
from code.helper.data import *
from multiprocessing import Process
from code.helper.augment import *
import code.data.config as config

path = config.path
dcfg = config.Default_cfg
dweights = config.Default_weights

def cv2_load_net(model_weights, model_config):
    # Load the pre-trained YOLO model and corresponding classes
    net = cv2.dnn.readNet(model_weights, model_config)
    return net

def detect_objects(image_path, net, classes):
    '''Detects objects in an image using the given YOLO net
    args:
        image_path: path to image
        net: YOLO net

    returns: list of [image_path, class_id, top_x, top_y, bottom_x, bottom_y, confidence]
    '''
    # Load image and get its dimensions
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    # Create a blob from the image and perform a forward pass
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    detections = net.forward(output_layers)
    image_name = os.path.basename(image_path)
    image_name = image_name[:-4]
    # Process the detections
    results = []
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust confidence threshold as needed
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                box_width = int(obj[2] * width)
                box_height = int(obj[3] * height)
                top_x, top_y, bottom_x, bottom_y = convert_yolo_to_voc(center_x, center_y, box_width, box_height)
                if top_x <= 6:
                    pass
                elif top_x >= 410:
                    pass
                else:
                    if top_y <= 6:
                        pass
                    elif top_y >= 410:
                        pass
                    else:
                        if bottom_x <= 6:
                            pass
                        elif bottom_x >= 410:
                            pass
                        else:
                            if bottom_y <= 6:
                                pass
                            elif bottom_y >= 410:
                                pass
                            else:
                                results.append([
                                    image_name,
                                    class_id,
                                    top_x,
                                    top_y,
                                    bottom_x,
                                    bottom_y,
                                    confidence
                                ])
    return results

def test_folder_cv2_DNN(folder_path, net, file_name, classes):
    # Specify the image file path
    image_paths = glob.glob(folder_path + "/*.jpg")

    # Perform object detection
    detections = []
    for image_path in tqdm.tqdm(image_paths, desc="Detecting objects"):
        detections += detect_objects(image_path, net, classes)
    file = open(file_name, "w")
    # Print the results
    for detection in detections:
        image_name, class_id, top_x, top_y, bottom_x, bottom_y, confidence = detection
        file.write(str(image_name) + ' ' + str(class_id) + ' ' + str(top_x) + ' ' + str(top_y) + ' ' + str(bottom_x) + ' ' + str(bottom_y) + ' ' + str(confidence) + '\n')
    file.close()    
    os.system("sed -i \'/^$/d\' " + file_name)

def train_easy(obj_data=path + "obj.data", cfg=dcfg, model=dweights, args=" -mjpeg_port 8090 -clear -dont_show"):
    '''Trains a model with the given parameters
    obj_data: path to obj.data file
    cfg: path to cfg file
    model: path to starting weights
    args: additional arguments to pass to darknet
    '''
    os.system("darknet detector train " + obj_data + ' ' + cfg + ' ' + model + ' ' + args )

def train_fancy(dir="", upper_range=10000, model=dweights, args=" -mjpeg_port 8090 -clear -dont_show", test=False):
    '''Trains a model with the given parameters
    dir: path to directory containing obj.data, cfg, and test folder
    upper_range: number of epochs to train for
    model: path to starting weights
    args: additional arguments to pass to darknet

    PS: will need to have the cfg file configured to train for only 10 epochs at a time
    PPS: good luck!
    '''
    if dir == "":
        raise Exception("Please provide a directory")
    li = []
    epoch = 0
    warnings.filterwarnings("ignore")
    obj_data = dir + "obj.data"
    # cfg_10 = dir + "yolov4_10.cfg"
    for file in os.listdir(dir):
        if file.endswith(".cfg"):
            cfg_10 = dir + file
            version = file.split('.')[0]
            print("Using cfg file: " + cfg_10)
    backup = dir + "backup/"
    names = dir + "obj.names"
    if not os.path.exists(backup):
        os.makedirs(backup)
    temp = dir + "temp/"
    if not os.path.exists(temp):
        os.makedirs(temp)
    new_weights = model
    test_dir = dir + "test/"
    test_file = test_dir + "test.txt"
    temp_file = temp + "temp.txt"
    make_ground_truth(temp + 'gt.txt', test_dir)
    print("Using obj.data file: " + obj_data)
    print("Using obj.names file: " + names)
    print("Using backup directory: " + backup)
    print("Using temp directory: " + temp)
    print("Using test directory: " + test_dir)
    print('Using Yolo version: ' + version)
    print("Initiating training for " + str(upper_range) + " epochs")
    print("Using starting weights: " + new_weights)
    for i in tqdm.tqdm(range(0, upper_range, 10), desc="Training", unit="epochs"):
        os.system("darknet detector train " + obj_data + ' ' + cfg_10 + ' ' + new_weights + ' ' + args + '> /dev/null 2>&1')
        # if epoch is mutilple of 100
        if epoch % 100 == 0:
            for i in range(1, 10 ,1):
                i = i * 10
                for file in os.listdir(backup):
                    if file.endswith(f"{i}.weights"):
                            os.system('rm ' + backup + file)
        epoch += 10
        subprocess.run(['mv', backup + version + '_final.weights', backup + version + '_' + str(epoch) + '.weights'])
        new_weights = backup + version + '_' + str(epoch) + ".weights"
        if test == True:
            os.system("darknet detector test " + obj_data + " " + cfg_10 + " " + new_weights + " -dont_show -ext_output < " + test_file + " > " + temp_file + " 2>&1")
            import_results_neo(temp_file, temp + 'results_' + str(epoch) + '.txt', names)
            F1w, F1m, acc, precision_score_weighted, precision_score_macro, recall_score_weighted, recall_score_macro, fbeta05_score_weighted, fbeta05_score_macro, fbeta2_score_weighted, fbeta2_score_macro, = do_math(temp + 'gt.txt', temp + 'results_' + str(epoch) + '.txt', 'export_' + str(epoch), temp, False, names, False)
            li.append([epoch, F1w, F1m, acc, precision_score_weighted, precision_score_macro, recall_score_weighted, recall_score_macro, fbeta05_score_weighted, fbeta05_score_macro, fbeta2_score_weighted, fbeta2_score_macro])
            os.system("rm " + temp + "results_" + str(epoch) + ".txt")
        subprocess.run(['rm', backup + 'chart_' + version + str(epoch) + '.png'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Training complete. Epochs: " + str(epoch))
    if test == True:                
        df = pd.DataFrame(li, columns = ['Epoch', 'F1_score_weighted', 'F1_score_macro', 'Accuracy', 'Precision_score_weighted', 'Precision_score_macro', 'Recall_score_weighted', 'Recall_score_macro', 'Fbeta05_score_weighted', 'Fbeta05_score_macro', 'Fbeta2_score_weighted', 'Fbeta2_score_macro'])
        pd.DataFrame(df).to_csv(dir + 'output.csv', index=False)

def get_info(data_path, model_path, model_name, sava_annotations=False):
    data_path = check_full_path(data_path)
    model_path = check_full_path(model_path)
    cfg = model_path + 'yolov4_10.cfg'
    model_name = check_full_path(model_name)
    data = model_path + 'obj.data'
    names = model_path + 'obj.names'
    temp_path = data_path + 'temp/'
    if os.path.exists(temp_path) == True:
        pass
    else:
        os.mkdir(temp_path)
    try:
        os.remove(data_path + 'test.txt')
    except:
        pass
    prep(data_path, 'test.txt')
    os.system('darknet detector test ' + data + ' ' + cfg + ' ' + model_name + ' -dont_show -ext_output < ' + data_path + 'test.txt' + ' > ' + temp_path + 'result.txt 2>&1')
    # os.system('darknet detector test ' + data + ' ' + cfg + ' ' + model_name + ' -dont_show -ext_output < ' + data_path + 'test.txt' + ' > ' + temp_path + 'result.txt')
    results = open(temp_path + 'result.txt', 'r')
    lines = results.readlines()
    save = []
    cells = ('LYM:', 'MON:', 'NEU:', 'ERY:', 'PLT:', 'ECHY', 'WBC:', 'LYM-', 'MON-', 'NEU-', 'WBC-')
    for line in lines:
        if line[0:4] in cells:
            lin = re.split(':|%|t|w|h', line)
            save.append([lin[0], int(lin[1])])
        else:
            pass
    df = pd.DataFrame(save, columns=['Cell type', 'Confidence'])
    os.remove(data_path + 'test.txt')
    df.to_csv(data_path + 'results.csv', index=False)
    ery = df.loc[df['Cell type'] == 'ERY']
    echy = df.loc[df['Cell type'] == 'ECHY']
    plt = df.loc[df['Cell type'] == 'PLT']
    wbc = df.loc[df['Cell type'] == 'WBC']
    lym = df.loc[df['Cell type'] == 'LYM']
    mon = df.loc[df['Cell type'] == 'MON']
    neu = df.loc[df['Cell type'] == 'NEU']
    lyma = df.loc[df['Cell type'] == 'LYM-']
    mona = df.loc[df['Cell type'] == 'MON-']
    neua = df.loc[df['Cell type'] == 'NEU-']
    wbca = df.loc[df['Cell type'] == 'WBC-']
    if os.path.exists(os.getcwd() + '/report.txt') == True:
        os.remove(os.getcwd() + '/report.txt')
    with open(os.getcwd() + '/report.txt', 'x') as f:
        f.write('Counted ' + str(len(df)) + ' cells\n')
        f.write('Overall average confidence: ' + str(round(float(df['Confidence'].mean()), 2)) + '\n')
        if len(ery) != 0:
            f.write('Counted ' + str(len(ery)) + ' erythrocytes\n')
            f.write('Average confidence: ' + str(round(float(ery['Confidence'].mean()), 2)) + '\n')
        if len(echy) != 0:
            f.write('Counted ' + str(len(echy)) + ' echinocytes\n')
            f.write('Average confidence: ' + str(round(float(echy['Confidence'].mean()), 2)) + '\n')
        if len(plt) != 0:
            f.write('Counted ' + str(len(plt)) + ' platelets\n')
            f.write('Average confidence: ' + str(round(float(plt['Confidence'].mean()), 2)) + '\n')
        if len(wbc) != 0:
            f.write('Counted ' + str(len(wbc)) + ' white blood cells\n')
            f.write('Average confidence: ' + str(round(float(wbc['Confidence'].mean()), 2)) + '\n')
        if len(lym) != 0:
            f.write('Counted ' + str(len(lym)) + ' lymphocytes\n')
            f.write('Average confidence: ' + str(round(float(lym['Confidence'].mean()), 2)) + '\n')
        if len(mon) != 0:
            f.write('Counted ' + str(len(mon)) + ' monocytes\n')
            f.write('Average confidence: ' + str(round(float(mon['Confidence'].mean()), 2)) + '\n')
        if len(neu) != 0:
            f.write('Counted ' + str(len(neu)) + ' neutrophils\n')
            f.write('Average confidence: ' + str(round(float(neu['Confidence'].mean()), 2)) + '\n')
        if len(lyma) != 0:
            f.write('Counted ' + str(len(lyma)) + ' activated lymphocytes\n')
            f.write('Average confidence: ' + str(round(float(lyma['Confidence'].mean()), 2)) + '\n')
        if len(mona) != 0:
            f.write('Counted ' + str(len(mona)) + ' activated monocytes\n')
            f.write('Average confidence: ' + str(round(float(mona['Confidence'].mean()), 2)) + '\n')
        if len(neua) != 0:
            f.write('Counted ' + str(len(neua)) + ' activated neutrophils-\n')
            f.write('Average confidence: ' + str(round(float(neua['Confidence'].mean()), 2)) + '\n')
        if len(wbca) != 0:
            f.write('Counted ' + str(len(wbca)) + ' activated white blood cells-\n')
            f.write('Average confidence: ' + str(round(float(wbca['Confidence'].mean()), 2)) + '\n')
    if sava_annotations == True:
        import_and_filter_result_neo(temp_path + 'result.txt', temp_path + 'results.txt', names)
        check_all_annotations_for_duplicates(temp_path + 'results.txt')
        with open(temp_path + 'results.txt') as f:
            for line in f:
                item = line.split()
                mv = [float(item[2]), float(item[3]), float(item[4]), float(item[5])]
                mv = pascal_to_yolo(mv)
                with open(temp_path + item[0] + '.txt', 'a') as g:
                    g.write(str(item[1]) + ' ' + str(mv[0]) + ' ' + str(mv[1]) + ' ' + str(mv[2]) + ' ' + str(mv[3]) + '\n')
        for image in os.listdir(data_path):
            if image.endswith(".jpg"):
                shutil.move(data_path + image, temp_path + image)
        os.system('cp ' + names + ' ' + temp_path + 'classes.txt')
        remove_non_annotated(temp_path)
        cat_file('report.txt')
        os.system('mv Report* ' + temp_path)
    inference_report('results.txt', 'Report')  
    os.remove(temp_path + 'result.txt')


def test_fancy(path, outpout_name, choose_wights = True, weights = dweights):
    path = check_full_path(path)
    if choose_wights == True:
        weights = choose_weights(path)
    cfg = choose_cfg(path)
    data = path + 'obj.data'
    names = path + 'obj.names'
    temp_path = path + 'temp/'
    if os.path.exists(temp_path) == True:
        pass
    else:
        os.mkdir(temp_path)
    prep(path + 'test/', 'test.txt')
    os.system('darknet detector test ' + data + ' ' + cfg + ' ' + weights + ' -dont_show -ext_output < ' + path + 'test/test.txt' + ' > ' + temp_path + 'result.txt 2>&1')
    make_ground_truth(temp_path + 'gt.txt', path + 'test/')
    import_and_filter_result_neo(temp_path + 'result.txt', temp_path + 'results.txt', names)
    check_all_annotations_for_duplicates(temp_path + 'results.txt')
    try:
        plot_bbox_area(temp_path + 'gt.txt', temp_path + 'results.txt', outpout_name, path, names)
    except:
        pass
    do_math(temp_path + 'gt.txt', temp_path + 'results.txt', outpout_name, path, True, names, True)

def train_5_fold_validation(folder_with_all_data, save_path, upper_range=10000, model=dweights, args=" -mjpeg_port 8090 -clear -dont_show", test=False, augment=False, augments=()):
    folder_with_all_data = check_full_path(folder_with_all_data)
    save_path = check_full_path(save_path)
    split_to_X_folders(folder_with_all_data, folder_with_all_data, 5)
    prepare_cfg_v4(folder_with_all_data+ '/classes.txt', save_path, 10)
    for i in range(1, 6, 1):
        print(i)
        try:
            os.mkdir(save_path + str(i))
        except: 
            pass    
        if i == 1:
            n1, n2, n3, n4, n5 = 1, 2, 3, 4, 5
        elif i == 2:
            n1, n2, n3, n4, n5 = 2, 3, 4, 5, 1
        elif i == 3:
            n1, n2, n3, n4, n5 = 3, 4, 5, 1, 2
        elif i == 4:
            n1, n2, n3, n4, n5 = 4, 5, 1, 2, 3
        elif i == 5:
            n1, n2, n3, n4, n5 = 5, 1, 2, 3, 4
        print(folder_with_all_data + f'f{n1}')
        combine_three_folders(folder_with_all_data, save_path + f'{n1}/train/', n1, n2, n3)
        shutil.copytree(folder_with_all_data + f'f{n4}/', save_path + f'{n1}/valid/', dirs_exist_ok=True)
        shutil.copytree(folder_with_all_data + f'f{n5}/', save_path + f'{n1}/test/', dirs_exist_ok=True)
        shutil.copy(save_path + f'{n1}/train/classes.txt', save_path + f'{n1}/obj.names')   
        if augment == True:
            iterate_augment(save_path + f'{n1}/train/', augments, True) 
        if os.path.exists(save_path+ f'{n1}/train/') == True:
            remove_non_annotated(save_path + f'{n1}/train/')
            prep(save_path + f'{n1}/train/', 'train.txt')
        if os.path.exists(save_path+f'{n1}/test/') == True:
            remove_non_annotated(save_path + f'{n1}/test/')
            prep(save_path + f'{n1}/test/', 'test.txt')
        if os.path.exists(save_path+ f'{n1}/valid/') == True:
            remove_non_annotated(save_path + f'{n1}/valid/')
            prep(save_path + f'{n1}/valid/', 'valid.txt')
        make_obj_data(save_path+f'{n1}/', False)
        shutil.copy(save_path + 'yolov4_10.cfg', save_path + f'{n1}/yolov4_10_pass_{n1}.cfg')
        # train_fancy(save_path + f'{n1}/', upper_range, model, args, test)

def test_5_fold_validation(work_dir, save_name, epochs=250):
    # folder = choose_folder(work_dir)
    folder = check_full_path(work_dir)
    for i in range(1, 6, 1):
        test_fancy(folder + f'/{i}/', save_name + f'_{i}', False, folder + f'/{i}/backup/yolov4_10_pass_{i}_{epochs}.weights')

def cv2_test_fancy(path, outpout_name, choose_wights = True, weights = dweights):
    path = check_full_path(path)
    if choose_wights == True:
        weights = choose_weights(path)
    cfg = choose_cfg(path)
    names = path + 'obj.names'
    test_dir = path + 'test/'
    temp_path = path + 'temp/'
    if os.path.exists(temp_path) == True:
        pass
    else:
        os.mkdir(temp_path)
    with open(names, "r") as f:
        classes = [line.strip() for line in f]    
    net = cv2_load_net(weights, cfg)
    test_folder_cv2_DNN(test_dir, net, temp_path + 'results.txt', classes)
    make_ground_truth(temp_path + 'gt.txt', test_dir)
    check_all_annotations_for_duplicates(temp_path + 'results.txt')
    tick = time.time()
    plot_bbox_area(temp_path + 'gt.txt', temp_path + 'results.txt', outpout_name, path, names)
    tack = time.time()
    do_math(temp_path + 'gt.txt', temp_path + 'results.txt', outpout_name, path, True, names, True)
    tock = time.time()
    print('Plotting: ' + str(tack - tick))
    print('Math: ' + str(tock - tack))
    

def test_5_fold_validation_cv2(work_dir, save_name, epochs=250):
    folder = check_full_path(work_dir)
    for i in range(1, 6, 1):
        cv2_test_fancy(folder + f'{i}/', save_name + f'_{i}', False, folder + f'{i}/backup/yolov4_10_pass_{i}_{epochs}.weights')

def multithread_test_5_fold_validation_cv2(work_dir, save_name, epochs=250):
    folder = check_full_path(work_dir)
    thread_list = []
    proc = os.cpu_count() - 1
    for i in range(1, 6, 1):
        thread = threading.Thread(target=cv2_test_fancy, args=(folder + f'{i}/', save_name + f'_{i}', False, folder + f'{i}/backup/yolov4_10_pass_{i}_{epochs}.weights'))
        thread_list.append(thread)
    while len(thread_list) > 0:
        for i in range(proc):
            try:
                thread_list[i].start()
                print('Started thread ' + str(i))
            except:
                pass
        for i in range(proc):
            try:
                thread_list[i].join()
                print('Joined thread ' + str(i))
            except:
                pass
        thread_list = thread_list[proc:]    

def multiprocess_test_5_fold_valdiation_cv2(workdir, save_name, epochs):
    folder = check_full_path(workdir)
    proc = os.cpu_count() - 1
    process_list = []
    for i in range(1,6,1):
        p = Process(target=cv2_test_fancy, args=(folder + f'{i}/', save_name + f'_{i}', False, folder + f'{i}/backup/yolov4_10_pass_{i}_{epochs}.weights'))
        process_list.append(p)
    [process.start() for process in process_list]
    [process.join() for process in process_list]

def test_training_epochs(work_dir, save_name):
    folder = check_full_path(work_dir)
    backup = folder + 'backup/'
    cfg = choose_cfg(folder)
    obj_data = folder + 'obj.data'
    temp = folder + 'temp/'
    if os.path.exists(temp) == True:
        pass
    else:
        os.mkdir(temp)
    names = folder + 'obj.names'
    test_weights = []
    li = []
    test_dir = folder + "test/"
    for file in os.listdir(backup):
        if file.endswith(".weights"):
            test_weights.append(file)
    for file in test_weights:
            name = file.split('_')[-1][:-8]
            net = cv2_load_net(backup + file, cfg)
            test_folder_cv2_DNN(test_dir, net, temp + 'results.txt', names)
            F1w, F1m, acc, precision_score_weighted, precision_score_macro, recall_score_weighted, recall_score_macro, fbeta05_score_weighted, fbeta05_score_macro, fbeta2_score_weighted, fbeta2_score_macro, = do_math(temp + 'gt.txt', temp + 'results_' + str(name) + '.txt', 'export_' + str(name), temp, False, names, False)
            li.append([name, F1w, F1m, acc, precision_score_weighted, precision_score_macro, recall_score_weighted, recall_score_macro, fbeta05_score_weighted, fbeta05_score_macro, fbeta2_score_weighted, fbeta2_score_macro])
    df = pd.DataFrame(li, columns = ['Epoch', 'F1_score_weighted', 'F1_score_macro', 'Accuracy', 'Precision_score_weighted', 'Precision_score_macro', 'Recall_score_weighted', 'Recall_score_macro', 'Fbeta05_score_weighted', 'Fbeta05_score_macro', 'Fbeta2_score_weighted', 'Fbeta2_score_macro'])
    pd.DataFrame(df).to_csv(folder + 'output.csv', index=False)
    make_training_graphs(folder + 'output.csv', folder)    