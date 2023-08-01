import os
import tqdm
import subprocess
import warnings
import pandas as pd
from code.helper.annotations import *
from code.helper.reports import *

def train_easy(obj_data="/home/as-hunt/Etra-Space/white-thirds/obj.data", cfg="/home/as-hunt/Etra-Space/white-thirds/yolov4.cfg", model="/home/as-hunt/Etra-Space/cfg/yolov4.conv.137", args=" -mjpeg_port 8090 -clear -dont_show"):
    '''Trains a model with the given parameters
    obj_data: path to obj.data file
    cfg: path to cfg file
    model: path to starting weights
    args: additional arguments to pass to darknet
    '''
    os.system("darknet detector train " + obj_data + ' ' + cfg + ' ' + model + ' ' + args )

def train_fancy(dir="/home/as-hunt/Etra-Space/white-thirds/", upper_range=10000, model="/home/as-hunt/Etra-Space/cfg/yolov4.conv.137", args=" -mjpeg_port 8090 -clear -dont_show"):
    '''Trains a model with the given parameters
    dir: path to directory containing obj.data, cfg, and test folder
    upper_range: number of epochs to train for
    model: path to starting weights
    args: additional arguments to pass to darknet
    
    PS: will need to have the cfg file configured to train for only 10 epochs at a time
    PPS: good luck!
    '''
    li = []
    epoch = 0
    warnings.filterwarnings("ignore")
    obj_data = dir + "obj.data"
    cfg_10 = dir + "yolov4_10.cfg"
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
    print("Initiating training for " + str(upper_range) + " epochs")
    print("Using starting weights: " + new_weights)
    for i in tqdm(range(0, upper_range, 10), desc="Training", unit="epochs"):
        os.system("darknet detector train " + obj_data + ' ' + cfg_10 + ' ' + new_weights + ' ' + args + '> /dev/null 2>&1')
        epoch = i + 10
        subprocess.run(['mv', backup + 'yolov4_10_final.weights', backup + 'yolov4_' + str(epoch) + '.weights'])
        new_weights = backup + "yolov4_" + str(epoch) + ".weights"
        os.system("darknet detector test " + obj_data + " " + cfg_10 + " " + new_weights + " -dont_show -ext_output < " + test_file + " > " + temp_file + " 2>&1")
        import_results_neo(temp_file, temp + 'results_' + str(epoch) + '.txt', names)
        F1w, F1m, acc, precision_score_weighted, precision_score_macro, recall_score_weighted, recall_score_macro, fbeta05_score_weighted, fbeta05_score_macro, fbeta2_score_weighted, fbeta2_score_macro, = do_math(temp + 'gt.txt', temp + 'results_' + str(epoch) + '.txt', 'export_' + str(epoch), False, ['ECHY', 'ERY', 'LYM', 'MON', 'NEU', 'PLT'], False)
        li.append([epoch, F1w, F1m, acc, precision_score_weighted, precision_score_macro, recall_score_weighted, recall_score_macro, fbeta05_score_weighted, fbeta05_score_macro, fbeta2_score_weighted, fbeta2_score_macro])
        os.system("rm " + temp + "results_" + str(epoch) + ".txt")
        if (epoch-10) % 50 == 0:
            pass
        else:
            subprocess.run(['rm', backup + 'yolov4_' + str(epoch - 10) + '.weights'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['rm', backup + 'chart_yolov4_' + str(epoch) + '.png'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    df = pd.DataFrame(li, columns = ['Epoch', 'F1_score_weighted', 'F1_score_macro', 'Accuracy', 'Precision_score_weighted', 'Precision_score_macro', 'Recall_score_weighted', 'Recall_score_macro', 'Fbeta05_score_weighted', 'Fbeta05_score_macro', 'Fbeta2_score_weighted', 'Fbeta2_score_macro'])    
    print("Training complete. Epochs: " + str(epoch))    
    pd.DataFrame(df).to_csv(dir + 'output.csv', index=False)

def get_info(data_path, model_path, model_name, output_report, sava_annotations=False):
    cfg = model_path + 'yolov4_10.cfg'
    weights = model_path + 'backup/' + model_name
    data = model_path + 'obj.data'
    names = model_path + 'obj.names'
    temp_path = data_path + 'temp/'
    if os.path.exists(temp_path) == True:
        pass
    else:
        os.mkdir(temp_path)
    if os.path.exists(data_path + 'test.txt') == True:
        os.remove(data_path + 'test.txt')
    else:
        filoo = open(data_path + 'test.txt', 'w')
        for image in os.listdir(data_path):
            if image.endswith(".jpg"):
                filoo.write(data_path + image + "\n")
        filoo.close()
    os.system('darknet detector test ' + data + ' ' + cfg + ' ' + weights + ' -dont_show -ext_output < ' + data_path + 'test.txt' + ' > ' + temp_path + 'result.txt 2>&1')
    results = open(temp_path + 'result.txt', 'r')
    lines = results.readlines()
    save = []
    cells = ('LYM:', 'MON:', 'NEU:', 'ERY:', 'PLT:', 'ECHY', 'WBC:')
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
    with open(output_report) as f:
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
    if sava_annotations == True:
        import_and_filter_result_neo(temp_path + 'result.txt', temp_path + 'results.txt', names)   
        with open(temp_path + 'results.txt') as f:
            for line in f:
                item = line.split()
                mv = [float(item[2]), float(item[3]), float(item[4]), float(item[5])]
                mv = [i / 416 for i in mv]
                with open(temp_path + item[0] + '.txt', 'a') as g:
                    g.write(str(item[1]) + ' ' + str(mv[0]) + ' ' + str(mv[1]) + ' ' + str(mv[2]) + ' ' + str(mv[3]) + '\n')
    os.remove(temp_path + 'results.txt')           
    os.remove(temp_path + 'result.txt')