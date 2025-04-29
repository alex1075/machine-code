import pandas as pd
import seaborn as sns
import numpy as np
import tqdm
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, fbeta_score
import matplotlib.pyplot as plt
from code.helper.annotations import *
from code.helper.utils import *
import multiprocessing
from functools import partial

def count_classes_file(test_file='/home/as-hunt/Etra-Space/new_data_sidless/gt.txt', chart=False, chart_name='chart.png', labs=['1', '2', '3']):
    class_1 = 0
    class_2 = 0
    class_3 = 0
    class_4 = 0
    class_5 = 0
    class_6 = 0
    count = 0
    annot = open(test_file, 'r+')
    for line in annot:
       lin = re.split(' ', line)
       classes = lin[1]
       if classes == '0':
          class_1 += 1
       elif classes == '1':
          class_2 += 1
       elif classes == '2':
          class_3 += 1
       elif classes == '3':
          class_4 += 1
       elif classes == '4':
          class_5 += 1
       elif classes == '5':
          class_6 += 1
    if chart == True:
        labels = labs
        plt.figure(figsize = (10,7))
        plt.title(chart_name[:-4])
        if len(labels) == 2:
            count = [class_1, class_2]
            fig, ax = plt.subplots()
            ax.pie(count, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        elif len(labels) == 3:
            count = [class_1, class_2, class_3]
            fig, ax = plt.subplots()
            ax.pie(count, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        elif len(labels) == 4:
            count = [class_1, class_2, class_3, class_4]
            fig, ax = plt.subplots()
            ax.pie(count, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        elif len(labels) == 5:
            count = [class_1, class_2, class_3, class_4, class_5]
            fig, ax = plt.subplots()
            ax.pie(count, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        elif len(labels) == 6:
            count = [class_1, class_2, class_3, class_4, class_5, class_6]
            fig, ax = plt.subplots()
            ax.pie(count, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.savefig(chart_name, bbox_inches='tight')    

@timing
def plot_bbox_area(gt_file, pd_file, save_name='areas', path='/home/as-hunt/', obj_name='/home/as-hunt/Etra-Space/white-thirds/obj.names'):
    '''Plots the areas of the bounding boxes in the ground truth and prediction from txt summary files'''
    names = []
    values = []
    gtchaart = []
    pdchaart = []
    areas = []
    gt_array = []
    pd_array = []
    dfp = []
    classesp = []
    classesg = []
    combined = []
    tagp = []
    tagg = []
    ious = []
    dfg = []
    temp = []
    target_names = []
    with open(obj_name, 'r') as f:
        lines = (line.rstrip() for line in f)
        lines = list(line for line in lines if line) # Non-blank lines in a list
        for line in lines:
            # print(line)
            temp.append(line)      
    for item in temp:
        if item == 'ECHY':
            target_names.append('Echinocytes')
        elif item == 'ERY':
            target_names.append('Erythrocyte')
        elif item == 'LYM':
            target_names.append('Lymphocyte')
        elif item == 'MON':
            target_names.append('Monocyte')
        elif item == 'NEU':
            target_names.append('Neutrophil')
        elif item == 'PLT':
            target_names.append('Platelet')
        elif item == 'WBC':
            target_names.append('White Blood Cell')          
        elif item == 'CTRL':
            target_names.append('Control')
        elif item == 'PHA':
            target_names.append('PHA')
        elif item == 'LYM-A':
            target_names.append('Lymphocyte-Activated')
        elif item == 'MON-A':
            target_names.append('Monocyte-Activated')
        elif item == 'NEU-A':
            target_names.append('Neutrophil-Activated')
    target_names.sort()     
    listed = open(pd_file, 'r')
    losted = open(gt_file, 'r')
    print('Plotting areas.')
    for line in listed:
        if line == '' or line == '\n' or line == ' \n':
            pass
        else:
            li = line.split(' ')
            try:
                name = li[0]
                classes = li[1]
                bbox = [int(li[2]), int(li[3]), int(li[4]), int(li[5])]
                confidence = li[6]
                pd_array.append([name, bbox, target_names[int(classes)], confidence])
            except:
                print(li)
                raise Exception('Error in plotting areas. Check the format of the prediction file.')
    for lune in losted:
        lu = lune.split(' ')
        nome = lu[0]
        clisses = lu[1]
        bbax = [int(lu[2]), int(lu[3]), int(lu[4]), int(lu[5])]
        gt_array.append([nome, bbax, target_names[int(clisses)]])
    for item in tqdm.tqdm(pd_array, unit='pd bbox'):
        name = item[0]
        bbox = item[1]
        classes = item[2]
        confidence = item[3]
        for thing in gt_array:
            nome = thing[0]
            bbax = thing[1]
            clisses = thing[2]
            if name in thing[0]:
                place = gt_array.index(thing)
                if iou(bbax, bbox) >= 0.5:
                    pdchaart.append([classes, (abs(int(bbox[2]) - int(bbox[0])) * abs(int(bbox[3]) - int(bbox[1])))])
                    gtchaart.append([clisses, (abs(int(bbax[2]) - int(bbax[0])) * abs(int(bbax[3]) - int(bbax[1])))])
                    classesp.append(classes)
                    classesg.append(clisses)    
                    # combined.append([(abs(int(bbox[2]) - int(bbox[0])) * abs(int(bbox[3]) - int(bbox[1]))), (abs(int(bbax[2]) - int(bbax[0])) * abs(int(bbax[3]) - int(bbax[1])))])     
                    dfp.append(float(abs(int(bbox[2]) - int(bbox[0])) * abs(int(bbox[3]) - int(bbox[1]))))
                    dfg.append(float(abs(int(bbax[2]) - int(bbax[0])) * abs(int(bbax[3]) - int(bbax[1]))))
                    tagp.append('PD')
                    tagg.append('GT')
                    if classes == clisses:
                        match = True
                        combined.append([(abs(int(bbax[2]) - int(bbax[0])) * abs(int(bbax[3]) - int(bbax[1]))), (abs(int(bbox[2]) - int(bbox[0])) * abs(int(bbox[3]) - int(bbox[1]))), classes, clisses, match, float(iou(bbax, bbox))])
                    else:   
                        match = False
                        combined.append([(abs(int(bbax[2]) - int(bbax[0])) * abs(int(bbax[3]) - int(bbax[1]))), (abs(int(bbox[2]) - int(bbox[0])) * abs(int(bbox[3]) - int(bbox[1]))), classes, clisses, match, float(iou(bbax, bbox))])
                    gt_array.pop(place)
    for item in areas:   
        names.append(item[0]) 
        values.append(item[1])
    fig, axs = plt.subplots(1, figsize=(9, 3), sharey=True)
    cl0 = []
    cl1 = []
    cl2 = []
    cl3 = []
    cl4 = []
    cl5 = []
    gcl0 = []
    gcl1 = []
    gcl2 = []
    gcl3 = []
    gcl4 = []
    gcl5 = []
    for item in pdchaart:
        if item[0] == '0':
            cl0.append(item[1])
        elif item[0] == '1':
            cl1.append(item[1])
        elif item[0] == '2':
            cl2.append(item[1])
        elif item[0] == '3':
            cl3.append(item[1])
        elif item[0] == '4':  
            cl4.append(item[1])
        elif item[0] == '5':
            cl5.append(item[1])          
    for item in gtchaart:
        if item[0] == '0':
            gcl0.append(item[1])
        elif item[0] == '1':
            gcl1.append(item[1])
        elif item[0] == '2':
            gcl2.append(item[1])
        elif item[0] == '3':
            gcl3.append(item[1])
        elif item[0] == '4':  
            gcl4.append(item[1])
        elif item[0] == '5':
            gcl5.append(item[1])    
    fig, axs = plt.subplots(2, 2)        
    fig.set_size_inches(16, 10)   
    df = pd.DataFrame({'Class':classesp, 'Area':dfp, 'Dataset':tagp}, columns=["Class", "Area", "Dataset"])
    for i in range(len(classesg)):
        new_row = {'Class': classesg[i], 'Area': dfg[i], 'Dataset': tagg[i]}
        df = df._append(new_row, ignore_index=True)
    sns.violinplot(data=df, cut=0, x='Class', y='Area', inner='box', scale='count', hue="Dataset", split=True, ax=axs[0, 0])
    axs[0, 0].set_title('Bbox Area Plotting per Class')
    du = pd.DataFrame(combined, columns=["x", "y", 'PD_class', 'GT_class', 'Match', 'IoU'])
    sns.scatterplot(data=du, x="x", y="y", ax=axs[1, 0], hue='Match', palette=["Red", "Blue",])
    axs[1, 0].set_title('Ground Truth Bbox by Predicted Bbox Areas coloured by Match of Classes')
    axs[1, 0].set(xlabel='Ground Truth Areas (pixels)', ylabel='Predicted Areas (pixels)')
    sns.scatterplot(data=du, x="x", y="y", ax=axs[0,1], hue='PD_class', palette=["Red", "Blue", "Green", "Purple", "Yellow", "Cyan"])
    axs[0, 1].set_title('Ground Truth Bbox by Predicted Bbox Areas coloured by Prediction Classes')
    axs[0, 1].set(xlabel='Ground Truth Areas (pixels)', ylabel='Predicted Areas (pixels)')
    sns.scatterplot(data=du, x="x", y="y", ax=axs[1, 1], hue='GT_class', palette=["Red", "Blue", "Green", "Purple", "Yellow", "Cyan"])
    axs[1, 1].set_title('Ground Truth Bbox by Predicted Bbox Areas coloured by Ground Truth Classes')
    axs[1, 1].set(xlabel='Ground Truth Areas (pixels)', ylabel='Predicted Areas (pixels)')
    plt.savefig(path + save_name + '_details.png', bbox_inches='tight')
    plt.clf()
    plt.cla()   

    
def export_errors(gt_file, pd_file, save_name='Error_', save_path='/home/as-hunt/', path2='/home/as-hunt/Etra-Space/white-thirds/test/'):
    '''Plots images with bounding boxes of the ground truth and prediction from txt summary files in one document for comparison'''
    path = save_path
    names = []
    values = []
    gtchaart = []
    pdchaart = []
    areas = []
    gt_array = []
    pd_array = []
    dfp = []
    classesp = []
    classesg = []
    combined = []
    tagp = []
    tagg = []
    ious = []
    dfg = []
    listed = open(pd_file, 'r')
    losted = open(gt_file, 'r')
    for line in listed:
        li = line.split(' ')
        name = li[0]
        classes = li[1]
        bbox = [int(li[2]), int(li[3]), int(li[4]), int(li[5])]
        confidence = li[6]
        pd_array.append([name, bbox, classes, confidence])
    for lune in losted:
        lu = lune.split(' ')
        nome = lu[0]
        clisses = lu[1]
        bbax = [int(lu[2]), int(lu[3]), int(lu[4]), int(lu[5])]
        gt_array.append([nome, bbax, clisses])
    for item in pd_array:
        name = item[0]
        bbox = item[1]
        classes = item[2]
        confidence = item[3]
        for thing in gt_array:
            nome = thing[0]
            bbax = thing[1]
            clisses = thing[2]
            if name in thing[0]:
                place = gt_array.index(thing)
                if iou(bbax, bbox) >= 0.5:
                    gt_image = path2 + name + '.jpg'
                    pd_image = path2 + nome + '.jpg'
                    labelled_gt_image = add_bbox(gt_image, bbax, clisses)
                    labelled_pd_image = add_bbox(pd_image, bbox, classes)
                    fig, axs = plt.subplots(1, 2)
                    axs[0].imshow(labelled_gt_image)
                    axs[0].set_title('Ground Truth')
                    axs[1].imshow(labelled_pd_image)
                    axs[1].set_title('Prediction')
                    plt.figtext(0.20, 0.15, 'Red - Lymphocyte, Green - Monocyte Blue - Neutrophil')
                    if classes == clisses:
                       save_name = path + 'Match_' + name + '.png'
                    else:   
                        save_name = path + 'Error_' + name + '.png'
                    plt.savefig(save_name , bbox_inches='tight')
                    gt_array.pop(place)     

def make_training_graphs(csv_file="/home/as-hunt/Etra-Space/white-thirds/output.csv", dir="/home/as-hunt/Etra-Space/white-thirds/"):
    '''This function takes in a csv file and outputs graphs of the training metrics
    _____________________________________________________________
    Args:
    
    csv_file: path to the csv file
    dir: path to save the output files
    ______________________________________________________________
    '''
    df = pd.read_csv(csv_file)
    plt.clf()
    plt.cla()   
    fig, axs = plt.subplots(2, 2)        
    fig.set_size_inches(16, 10)
    sns.lineplot(x="Epoch", y="Accuracy", data=df, ax=axs[0, 0])
    axs[0, 0].set_title("Accuracy over epochs")
    axs[0, 0].set(xlabel='Epoch', ylabel='Accuracy (in %)')

    sns.lineplot(x="Epoch", y="F1_score_weighted", data=df, ax=axs[0, 1])
    axs[0, 1].set_title("F1 score over epochs")
    axs[0, 1].set(xlabel='Epoch', ylabel='F1 score (weighted, in %)')

    sns.lineplot(x="Epoch", y="Precision_score_weighted", data=df, ax=axs[1, 0])
    axs[1, 0].set_title("Precision score over epochs")
    axs[1, 0].set(xlabel='Epoch', ylabel='Precision score (weighted, in %)')

    sns.lineplot(x="Epoch", y="Recall_score_weighted", data=df, ax=axs[1, 1])
    axs[1, 1].set_title("Recall score over epochs")
    axs[1, 1].set(xlabel='Epoch', ylabel='Recall score (weighted, in %)')

    plt.savefig(dir + "Training output.png", bbox_inches='tight')
    plt.clf()
    fig, axs = plt.subplots(1, 1)
    sns.set_theme(style="whitegrid")
    dfm=df.melt('Epoch', var_name='cols',  value_name='vals')
    fig.set_size_inches(16, 10)
    sns.lineplot(data=dfm, x="Epoch", y="vals", hue='cols')
    plt.savefig(dir + "Training output together.png", bbox_inches='tight')

def inference_report(repot_txt, save_name='areas.png'):
    '''Plots the areas of the bounding boxes in the ground truth and prediction from txt summary files'''
    plt.clf()
    plt.cla()   
    pdchaart = []
    confi = []
    pd_array = []
    scatter = []
    dfp = []
    classesp = []
    listed = open(repot_txt, 'r')
    for line in listed:
        li = line.split(' ')
        name = li[0]
        classes = li[1]
        bbox = [int(li[2]), int(li[3]), int(li[4]), int(li[5])]
        confidence = li[6]
        pd_array.append([name, bbox, classes, confidence])
    for item in pd_array:
        name = item[0]
        bbox = item[1]
        classes = item[2]
        confidence = item[3]
        pdchaart.append([classes, (abs(int(bbox[2]) - int(bbox[0])) * abs(int(bbox[3]) - int(bbox[1])))])
        classesp.append(classes)
        confi.append(confidence)
        dfp.append(float(abs(int(bbox[2]) - int(bbox[0])) * abs(int(bbox[3]) - int(bbox[1]))))
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        scatter.append([int(width), int(height), int(classes), float(confidence)])
    fig, axs = plt.subplots(3)        
    fig.set_size_inches(10, 16)
    confi = [float(i) for i in confi]
    df = pd.DataFrame({'Class':classesp, 'Area':dfp}, columns=["Class", "Area"])
    sns.violinplot(data=df, cut=0, x='Class', y='Area', inner='box', scale='count', split=True, ax=axs[0], order=["0", "1", "2", "3", "4", "5"], palette=["Red", "Blue", "Green", "Purple", "Yellow", "Cyan"])
    axs[0].set_title('Bbox Area Plotting per Class')
    axs[0].set(xlabel='Class', ylabel='Bbox Areas (pixels)')
    classesp = [int(i) for i in classesp]
    du = pd.DataFrame({'Class':classesp, 'Area':dfp, 'Confidence':confi}, columns=["Class", "Area", "Confidence"])
    sns.scatterplot(data=du, x="Area", y="Confidence", ax=axs[1], hue='Class', palette=["Red", "Blue", "Green", "Purple", "Yellow", "Cyan"])
    axs[1].set_title('Confidence by Bbox Areas coloured by Prediction Classes')
    axs[1].set(xlabel='Bbox Areas (pixels)', ylabel='Confidence')
    da = pd.DataFrame(scatter, columns=["Width", "Height", "Class", "Confidence"])
    sns.scatterplot(data=da, x="Width", y="Height",ax=axs[2], hue='Class',)
    sns.histplot(data=da, x='Width', y='Height', ax=axs[2], bins=50, pthresh=.1, cmap="mako")
    sns.kdeplot(data=da, x='Width', y='Height', ax=axs[2], levels=5, color="r", linewidths=1)
    axs[2].set_title('Scatterplot')
    axs[2].set(xlabel='Width', ylabel='Height')
    plt.savefig(save_name+'_details.png', bbox_inches='tight')
    plt.clf()
    plt.cla()   

def process_item(item, gt_array, pdchaart, gtchaart, classesp, classesg, dfp, dfg, tagp, tagg, combined, target_names):
    name, bbox, classes, confidence = item
    for thing in gt_array:
        nome, bbax, clisses = thing
        if name in thing[0]:
            place = gt_array.index(thing)
            if iou(bbax, bbox) >= 0.5:
                pdchaart.append([classes, (abs(int(bbox[2]) - int(bbox[0])) * abs(int(bbox[3]) - int(bbox[1])))])
                gtchaart.append([clisses, (abs(int(bbax[2]) - int(bbax[0])) * abs(int(bbax[3]) - int(bbax[1])))])
                classesp.append(classes)
                classesg.append(clisses)
                dfp.append(float(abs(int(bbox[2]) - int(bbox[0])) * abs(int(bbox[3]) - int(bbox[1]))))
                dfg.append(float(abs(int(bbax[2]) - int(bbax[0])) * abs(int(bbax[3]) - int(bbax[1]))))
                tagp.append('PD')
                tagg.append('GT')
                if classes == clisses:
                    match = True
                    combined.append([(abs(int(bbax[2]) - int(bbax[0])) * abs(int(bbax[3]) - int(bbax[1]))), (abs(int(bbox[2]) - int(bbox[0])) * abs(int(bbox[3]) - int(bbox[1]))), classes, clisses, match, float(iou(bbax, bbox))])
                else:   
                    match = False
                    combined.append([(abs(int(bbax[2]) - int(bbax[0])) * abs(int(bbax[3]) - int(bbax[1]))), (abs(int(bbox[2]) - int(bbox[0])) * abs(int(bbox[3]) - int(bbox[1]))), classes, clisses, match, float(iou(bbax, bbox))])
                gt_array.pop(place)

def parallel_process_items(pd_array, gt_array, target_names):
    with multiprocessing.Manager() as manager:
        pdchaart = manager.list()
        gtchaart = manager.list()
        classesp = manager.list()
        classesg = manager.list()
        dfp = manager.list()
        dfg = manager.list()
        tagp = manager.list()
        tagg = manager.list()
        combined = manager.list()

        pool = multiprocessing.Pool()
        func = partial(process_item, gt_array=gt_array, pdchaart=pdchaart, gtchaart=gtchaart, classesp=classesp, classesg=classesg, dfp=dfp, dfg=dfg, tagp=tagp, tagg=tagg, combined=combined, target_names=target_names)
        for _ in tqdm.tqdm(pool.imap_unordered(func, pd_array), total=len(pd_array), unit='pd bbox'):
            pass
        pool.close()
        pool.join()

        return pdchaart, gtchaart, classesp, classesg, dfp, dfg, tagp, tagg, combined



@timing
def mplot_bbox_area(gt_file, pd_file, save_name='areas', path='/home/as-hunt/', obj_name='/home/as-hunt/Etra-Space/white-thirds/obj.names'):
    '''Plots the areas of the bounding boxes in the ground truth and prediction from txt summary files'''
    names = []
    values = []
    gtchaart = []
    pdchaart = []
    areas = []
    gt_array = []
    pd_array = []
    dfp = []
    classesp = []
    classesg = []
    combined = []
    tagp = []
    tagg = []
    ious = []
    dfg = []
    temp = []
    target_names = []
    with open(obj_name, 'r') as f:
        lines = (line.rstrip() for line in f)
        lines = list(line for line in lines if line) # Non-blank lines in a list
        for line in lines:
            # print(line)
            temp.append(line)      
    for item in temp:
        if item == 'ECHY':
            target_names.append('Echinocytes')
        elif item == 'ERY':
            target_names.append('Erythrocyte')
        elif item == 'LYM':
            target_names.append('Lymphocyte')
        elif item == 'MON':
            target_names.append('Monocyte')
        elif item == 'NEU':
            target_names.append('Neutrophil')
        elif item == 'PLT':
            target_names.append('Platelet')
        elif item == 'WBC':
            target_names.append('White Blood Cell')          
        elif item == 'CTRL':
            target_names.append('Control')
        elif item == 'PHA':
            target_names.append('PHA')
        elif item == 'LYM-A':
            target_names.append('Lymphocyte-Activated')
        elif item == 'MON-A':
            target_names.append('Monocyte-Activated')
        elif item == 'NEU-A':
            target_names.append('Neutrophil-Activated')
    target_names.sort()     
    listed = open(pd_file, 'r')
    losted = open(gt_file, 'r')
    print('Plotting areas.')
    for line in listed:
        li = line.split(' ')
        name = li[0]
        classes = li[1]
        bbox = [int(li[2]), int(li[3]), int(li[4]), int(li[5])]
        confidence = li[6]
        pd_array.append([name, bbox, target_names[int(classes)], confidence])
    for lune in losted:
        lu = lune.split(' ')
        nome = lu[0]
        clisses = lu[1]
        bbax = [int(lu[2]), int(lu[3]), int(lu[4]), int(lu[5])]
        gt_array.append([nome, bbax, target_names[int(clisses)]])
    pdchaart, gtchaart, classesp, classesg, dfp, dfg, tagp, tagg, combined = parallel_process_items(pd_array, gt_array, target_names)
    for item in areas:   
        names.append(item[0]) 
        values.append(item[1])
    fig, axs = plt.subplots(1, figsize=(9, 3), sharey=True)
    cl0 = []
    cl1 = []
    cl2 = []
    cl3 = []
    cl4 = []
    cl5 = []
    gcl0 = []
    gcl1 = []
    gcl2 = []
    gcl3 = []
    gcl4 = []
    gcl5 = []
    for item in pdchaart:
        if item[0] == '0':
            cl0.append(item[1])
        elif item[0] == '1':
            cl1.append(item[1])
        elif item[0] == '2':
            cl2.append(item[1])
        elif item[0] == '3':
            cl3.append(item[1])
        elif item[0] == '4':  
            cl4.append(item[1])
        elif item[0] == '5':
            cl5.append(item[1])          
    for item in gtchaart:
        if item[0] == '0':
            gcl0.append(item[1])
        elif item[0] == '1':
            gcl1.append(item[1])
        elif item[0] == '2':
            gcl2.append(item[1])
        elif item[0] == '3':
            gcl3.append(item[1])
        elif item[0] == '4':  
            gcl4.append(item[1])
        elif item[0] == '5':
            gcl5.append(item[1])    
    fig, axs = plt.subplots(2, 2)        
    fig.set_size_inches(16, 10)   
    df = pd.DataFrame({'Class':classesp, 'Area':dfp, 'Dataset':tagp}, columns=["Class", "Area", "Dataset"])
    for i in range(len(classesg)):
        new_row = {'Class': classesg[i], 'Area': dfg[i], 'Dataset': tagg[i]}
        df = df._append(new_row, ignore_index=True)
    sns.violinplot(data=df, cut=0, x='Class', y='Area', inner='box', scale='count', hue="Dataset", split=True, ax=axs[0, 0])
    axs[0, 0].set_title('Bbox Area Plotting per Class')
    du = pd.DataFrame(combined, columns=["x", "y", 'PD_class', 'GT_class', 'Match', 'IoU'])
    sns.scatterplot(data=du, x="x", y="y", ax=axs[1, 0], hue='Match', palette=["Red", "Blue",])
    axs[1, 0].set_title('Ground Truth Bbox by Predicted Bbox Areas coloured by Match of Classes')
    axs[1, 0].set(xlabel='Ground Truth Areas (pixels)', ylabel='Predicted Areas (pixels)')
    sns.scatterplot(data=du, x="x", y="y", ax=axs[0,1], hue='PD_class', palette=["Red", "Blue", "Green", "Purple", "Yellow", "Cyan"])
    axs[0, 1].set_title('Ground Truth Bbox by Predicted Bbox Areas coloured by Prediction Classes')
    axs[0, 1].set(xlabel='Ground Truth Areas (pixels)', ylabel='Predicted Areas (pixels)')
    sns.scatterplot(data=du, x="x", y="y", ax=axs[1, 1], hue='GT_class', palette=["Red", "Blue", "Green", "Purple", "Yellow", "Cyan"])
    axs[1, 1].set_title('Ground Truth Bbox by Predicted Bbox Areas coloured by Ground Truth Classes')
    axs[1, 1].set(xlabel='Ground Truth Areas (pixels)', ylabel='Predicted Areas (pixels)')
    plt.savefig(path + save_name + '_details.png', bbox_inches='tight')
    plt.clf()
    plt.cla()

def calculate_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Automatically corrects coordinate ordering if necessary.

    Parameters
    ----------
    bb1 : list
        [x1, y1, x2, y2]
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : list
        [x1, y1, x2, y2]
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    # Make copies to avoid modifying the original arrays
    bb1 = bb1.copy()
    bb2 = bb2.copy()
    
    # Ensure coordinates are in the correct order (x1 <= x2, y1 <= y2)
    if bb1[0] > bb1[2]:
        bb1[0], bb1[2] = bb1[2], bb1[0]
    if bb1[1] > bb1[3]:
        bb1[1], bb1[3] = bb1[3], bb1[1]
    
    if bb2[0] > bb2[2]:
        bb2[0], bb2[2] = bb2[2], bb2[0]
    if bb2[1] > bb2[3]:
        bb2[1], bb2[3] = bb2[3], bb2[1]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box.
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    
    # Sanity check
    if not (0.0 <= iou <= 1.0):
        iou = max(0.0, min(1.0, iou))
    
    return iou

def is_near_edge(box, image_size=(416, 416), threshold=6):
    """
    Check if a bounding box is near the edge of an image.
    
    Parameters:
    -----------
    box : list
        [x1, y1, x2, y2] bounding box coordinates
    image_size : tuple, optional
        (width, height) of the image
    threshold : int, optional
        Distance threshold in pixels
        
    Returns:
    --------
    bool
        True if the box is near the edge, False otherwise
    """
    x1, y1, x2, y2 = box
    width, height = image_size
    
    # Check if any part of the box is within threshold pixels of any edge
    if (x1 < threshold or  # Left edge
        y1 < threshold or  # Top edge
        x2 > width - threshold or  # Right edge
        y2 > height - threshold):  # Bottom edge
        return True
    
    return False

def calculate_results(pd_file, gt_file, edge_threshold=6):
    # Load the predicted and ground truth detections
    pd_detections = np.loadtxt(pd_file, delimiter=" ", dtype=str)
    gt_detections = np.loadtxt(gt_file, delimiter=" ", dtype=str)
    
    # Filter out detections near the edge
    pd_detections_filtered = []
    for pd_detection in pd_detections:
        pd_image_name, pd_class, pd_x1, pd_y1, pd_x2, pd_y2 = pd_detection[:6]
        pd_box = [float(pd_x1), float(pd_y1), float(pd_x2), float(pd_y2)]
        if not is_near_edge(pd_box, threshold=edge_threshold):
            pd_detections_filtered.append(pd_detection)
    
    gt_detections_filtered = []
    for gt_detection in gt_detections:
        gt_image_name, gt_class, gt_x1, gt_y1, gt_x2, gt_y2 = gt_detection[:6]
        gt_box = [float(gt_x1), float(gt_y1), float(gt_x2), float(gt_y2)]
        if not is_near_edge(gt_box, threshold=edge_threshold):
            gt_detections_filtered.append(gt_detection)
    
    # Use filtered detections for calculations
    pd_detections = np.array(pd_detections_filtered)
    gt_detections = np.array(gt_detections_filtered)
    
    # Track which ground truths have been matched
    gt_matched = [False] * len(gt_detections)
    
    # Calculate the number of true positives and false positives
    tp, fp = 0, 0
    for pd_detection in tqdm.tqdm(pd_detections, leave=False):
        pd_image_name, pd_class, pd_x1, pd_y1, pd_x2, pd_y2 = pd_detection[:6]
        pd_box = [float(pd_x1), float(pd_y1), float(pd_x2), float(pd_y2)]
        
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_detection in enumerate(gt_detections):
            # Skip already matched ground truths
            if gt_matched[gt_idx]:
                continue
                
            gt_image_name, gt_class, gt_x1, gt_y1, gt_x2, gt_y2 = gt_detection[:6]
            gt_box = [float(gt_x1), float(gt_y1), float(gt_x2), float(gt_y2)]
            
            iou = calculate_iou(pd_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou > 0.5 and best_gt_idx >= 0:
            if pd_class == gt_detections[best_gt_idx][1]:
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                fp += 1
        else:
            fp += 1
    
    # False negatives are ground truths that weren't matched
    fn = len(gt_detections) - sum(gt_matched)
    
    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def calculate_results_per_class(pd_file, gt_file, edge_threshold=6):
    # Load the predicted and ground truth detections
    pd_detections = np.loadtxt(pd_file, delimiter=" ", dtype=str)
    gt_detections = np.loadtxt(gt_file, delimiter=" ", dtype=str)
    
    # Filter out detections near the edge
    pd_detections_filtered = []
    for pd_detection in pd_detections:
        pd_image_name, pd_class, pd_x1, pd_y1, pd_x2, pd_y2 = pd_detection[:6]
        pd_box = [float(pd_x1), float(pd_y1), float(pd_x2), float(pd_y2)]
        if not is_near_edge(pd_box, threshold=edge_threshold):
            pd_detections_filtered.append(pd_detection)
    
    gt_detections_filtered = []
    for gt_detection in gt_detections:
        gt_image_name, gt_class, gt_x1, gt_y1, gt_x2, gt_y2 = gt_detection[:6]
        gt_box = [float(gt_x1), float(gt_y1), float(gt_x2), float(gt_y2)]
        if not is_near_edge(gt_box, threshold=edge_threshold):
            gt_detections_filtered.append(gt_detection)
    
    # Use filtered detections for calculations
    pd_detections = np.array(pd_detections_filtered)
    gt_detections = np.array(gt_detections_filtered)
    
    # Track which ground truths have been matched
    gt_matched = [False] * len(gt_detections)
    
    # Calculate the number of true positives and false positives
    class_tp, class_fp = {}, {}
    for pd_detection in tqdm.tqdm(pd_detections, leave=False):
        pd_image_name, pd_class, pd_x1, pd_y1, pd_x2, pd_y2 = pd_detection[:6]
        pd_box = [float(pd_x1), float(pd_y1), float(pd_x2), float(pd_y2)]
        
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_detection in enumerate(gt_detections):
            # Skip already matched ground truths
            if gt_matched[gt_idx]:
                continue
                
            gt_image_name, gt_class, gt_x1, gt_y1, gt_x2, gt_y2 = gt_detection[:6]
            gt_box = [float(gt_x1), float(gt_y1), float(gt_x2), float(gt_y2)]
            
            iou = calculate_iou(pd_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou > 0.5 and best_gt_idx >= 0:
            if pd_class == gt_detections[best_gt_idx][1]:
                class_tp[pd_class] = class_tp.get(pd_class, 0) + 1
                gt_matched[best_gt_idx] = True
            else:
                class_fp[pd_class] = class_fp.get(pd_class, 0) + 1
        else:
            class_fp[pd_class] = class_fp.get(pd_class, 0) + 1
    
    # Calculate false negatives based on unmatched ground truths
    class_fn = {}
    for gt_idx, gt_detection in enumerate(gt_detections):
        if not gt_matched[gt_idx]:
            gt_class = gt_detection[1]
            class_fn[gt_class] = class_fn.get(gt_class, 0) + 1
    
    # Calculate precision, recall, and F1 score for each class
    class_metrics = {}
    for cls in tqdm.tqdm(set(class_tp.keys()).union(class_fp.keys()).union(class_fn.keys()), leave=False):
        tp = class_tp.get(cls, 0)
        fp = class_fp.get(cls, 0)
        fn = class_fn.get(cls, 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    return class_metrics

def create_improved_confusion_matrix(pd_file, gt_file, iou_threshold=0.5, edge_threshold=6):
    """
    Create an improved confusion matrix for object detection results with better
    handling of class matching.
    
    Parameters:
    -----------
    pd_file : str
        Path to the predicted detections file.
    gt_file : str
        Path to the ground truth detections file.
    iou_threshold : float, optional
        IoU threshold for matching detections.
    edge_threshold : int, optional
        Distance threshold in pixels for filtering objects near edges.
        
    Returns:
    --------
    tuple
        (confusion_matrix, row_labels, column_labels)
    """
    # Load the predicted and ground truth detections
    pd_detections = np.loadtxt(pd_file, delimiter=" ", dtype=str)
    gt_detections = np.loadtxt(gt_file, delimiter=" ", dtype=str)
    
    # Ensure arrays are properly shaped
    if pd_detections.ndim == 1:
        pd_detections = pd_detections.reshape(1, -1)
    if gt_detections.ndim == 1:
        gt_detections = gt_detections.reshape(1, -1)
    
    # Filter out detections near the edge
    pd_detections_filtered = []
    for pd_detection in pd_detections:
        pd_image_name, pd_class, pd_x1, pd_y1, pd_x2, pd_y2 = pd_detection[:6]
        pd_box = [float(pd_x1), float(pd_y1), float(pd_x2), float(pd_y2)]
        if not is_near_edge(pd_box, threshold=edge_threshold):
            pd_detections_filtered.append(pd_detection)
    
    gt_detections_filtered = []
    for gt_detection in gt_detections:
        gt_image_name, gt_class, gt_x1, gt_y1, gt_x2, gt_y2 = gt_detection[:6]
        gt_box = [float(gt_x1), float(gt_y1), float(gt_x2), float(gt_y2)]
        if not is_near_edge(gt_box, threshold=edge_threshold):
            gt_detections_filtered.append(gt_detection)
    
    # Use filtered detections for calculations
    pd_detections = np.array(pd_detections_filtered)
    gt_detections = np.array(gt_detections_filtered)
    
    # Get all unique classes
    gt_classes = sorted(list(set([det[1] for det in gt_detections])))
    pd_classes = sorted(list(set([det[1] for det in pd_detections])))
    all_classes = sorted(list(set(gt_classes).union(set(pd_classes))))
    
    # Create class to index mapping
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    
    # Initialize confusion matrix
    # Rows = GT classes, Columns = PD classes + "Non-detected"
    n_classes = len(all_classes)
    confusion_mat = np.zeros((n_classes, n_classes + 1), dtype=int)
    
    # Group gt detections by image for more efficient matching
    gt_by_image = {}
    for gt_idx, gt_detection in enumerate(gt_detections):
        img_name = gt_detection[0]
        if img_name not in gt_by_image:
            gt_by_image[img_name] = []
        gt_by_image[img_name].append((gt_idx, gt_detection))
    
    # Track matched ground truths
    gt_matched = [False] * len(gt_detections)
    
    # For each prediction, find best matching ground truth
    for pd_detection in tqdm.tqdm(pd_detections, leave=False):
        pd_image_name, pd_class, pd_x1, pd_y1, pd_x2, pd_y2 = pd_detection[:6]
        pd_box = [float(pd_x1), float(pd_y1), float(pd_x2), float(pd_y2)]
        pd_idx = class_to_idx[pd_class]
        
        best_iou = 0
        best_gt_idx = -1
        
        # Only check ground truths from the same image
        if pd_image_name in gt_by_image:
            for gt_idx, gt_detection in gt_by_image[pd_image_name]:
                # Skip already matched ground truths
                if gt_matched[gt_idx]:
                    continue
                    
                gt_class = gt_detection[1]
                gt_x1, gt_y1, gt_x2, gt_y2 = (float(gt_detection[2]), float(gt_detection[3]), 
                                             float(gt_detection[4]), float(gt_detection[5]))
                gt_box = [gt_x1, gt_y1, gt_x2, gt_y2]
                
                iou = calculate_iou(pd_box, gt_box)
                
                # Prioritize matching with same class
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        if best_iou > iou_threshold and best_gt_idx >= 0:
            # True detection with correct or incorrect class
            gt_class = gt_detections[best_gt_idx][1]
            gt_idx = class_to_idx[gt_class]
            confusion_mat[gt_idx][pd_idx] += 1
            gt_matched[best_gt_idx] = True
        else:
            # False positive without a matching ground truth
            # We could count this as a separate category, but for now we'll skip it
            pass
    
    # Process unmatched ground truths (non-detected objects)
    for gt_idx, gt_detection in enumerate(gt_detections):
        if not gt_matched[gt_idx]:
            gt_class = gt_detection[1]
            gt_idx = class_to_idx[gt_class]
            # Last column represents "Non-detected"
            confusion_mat[gt_idx][n_classes] += 1
    
    # Prepare labels
    row_labels = all_classes
    col_labels = all_classes + ["Non-detected"]
    
    return confusion_mat, row_labels, col_labels

def calculate_metrics_from_confusion_matrix(confusion_mat, row_labels):
    """
    Calculate precision, recall, and F1 score from a confusion matrix.
    
    Parameters:
    -----------
    confusion_mat : numpy.ndarray
        Confusion matrix with rows as true classes and columns as predicted classes + "Non-detected"
    row_labels : list
        Class labels corresponding to rows in the confusion matrix
        
    Returns:
    --------
    tuple
        (overall_metrics, per_class_metrics)
    """
    n_classes = len(row_labels)
    per_class_metrics = {}
    
    # Calculate metrics for each class
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for i, cls in enumerate(row_labels):
        # True positives: diagonal element (correct predictions)
        tp = confusion_mat[i][i]
        
        # False positives: sum of column i (excluding the diagonal element)
        fp = sum(confusion_mat[j][i] for j in range(n_classes) if j != i)
        
        # False negatives: sum of row i (excluding diagonal) + non-detected
        fn = sum(confusion_mat[i][j] for j in range(n_classes) if j != i) + confusion_mat[i][n_classes]
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        # Accumulate for macro average
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate overall metrics (macro average)
    overall_precision = sum(m['precision'] for m in per_class_metrics.values()) / n_classes
    overall_recall = sum(m['recall'] for m in per_class_metrics.values()) / n_classes
    overall_f1 = sum(m['f1_score'] for m in per_class_metrics.values()) / n_classes
    
    overall_metrics = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1
    }
    
    return overall_metrics, per_class_metrics

def print_metrics(overall_metrics, per_class_metrics):
    """Print metrics in a readable format."""
    print(f"Precision: {overall_metrics['precision']}, Recall: {overall_metrics['recall']}, F1 Score: {overall_metrics['f1_score']}")
    print(per_class_metrics)

def print_confusion_matrix(confusion_mat, row_labels, col_labels):
    """Print the confusion matrix in a readable format."""
    # Calculate column widths for better alignment
    col_width = max(max(len(str(x)) for x in row_labels), max(len(str(x)) for x in col_labels)) + 2
    
    # Print header row
    print(" " * col_width, end="")
    for label in col_labels:
        print(f"{label:{col_width}}", end="")
    print()
    
    # Print each row
    for i, label in enumerate(row_labels):
        print(f"{label:{col_width}}", end="")
        for j in range(len(col_labels)):
            print(f"{confusion_mat[i][j]:{col_width}}", end="")
        print()    

def do_math(gt_file, pd_file, title, path, save_txt=False, obj_name='/home/as-hunt/Etra-Space/white-thirds/obj.names', save_png=False, edge_threshold=6):
    '''This function takes in a ground truth file and a prediction file and returns AI metrics for the model
    using the calculate_results.py functions for more accurate detection matching
    
    _____________________________________________________________
    Args:

    gt_file: ground truth file
    pd_file: prediction file
    
    title: title of the output files
    path: path to save the output files
    
    save_txt: boolean, whether or not to save the text file
    obj_name: path to the obj.names file to use for the confusion matrix
    save_png: boolean, whether or not to save the png file
    edge_threshold: distance threshold in pixels for filtering objects near edges
    ______________________________________________________________
    '''
    plt.clf()
    plt.cla()   
    
    # Load class names
    target_names = []
    temp = []
    with open(obj_name, 'r') as f:
        lines = (line.rstrip() for line in f)
        lines = list(line for line in lines if line)  # Non-blank lines in a list
        for line in lines:
            temp.append(line)      
    
    for item in temp:
        if item == 'ECHY':
            target_names.append('Echinocytes')
        elif item == 'ERY':
            target_names.append('Erythrocyte')
        elif item == 'LYM':
            target_names.append('Lymphocyte')
        elif item == 'MON':
            target_names.append('Monocyte')
        elif item == 'NEU':
            target_names.append('Neutrophil')
        elif item == 'PLT':
            target_names.append('Platelet')
        elif item == 'WBC':
            target_names.append('White Blood Cell')          
        elif item == 'CTRL':
            target_names.append('Control')
        elif item == 'PHA':
            target_names.append('PHA')
        elif item == 'LYM-A':
            target_names.append('Lymphocyte-Activated')
        elif item == 'MON-A':
            target_names.append('Monocyte-Activated')
        elif item == 'NEU-A':
            target_names.append('Neutrophil-Activated')
    target_names.sort()
    
    # Calculate overall metrics using calculate_results.py functions
    print(f"Calculating metrics for {title}...")
    
    # Get overall precision, recall, and F1 score
    precision, recall, f1_score = calculate_results(pd_file, gt_file, edge_threshold)
    
    # Get per-class metrics
    class_metrics = calculate_results_per_class(pd_file, gt_file, edge_threshold)
    
    # Create confusion matrix with improved matching
    conf_mat, row_labels, col_labels = create_improved_confusion_matrix(pd_file, gt_file, 0.5, edge_threshold)
    
    # Calculate metrics from the confusion matrix for consistency
    overall_metrics, per_class_metrics = calculate_metrics_from_confusion_matrix(conf_mat, row_labels)
    
    # Format metrics for output
    F1m = overall_metrics['f1_score']
    F1w = f1_score
    precision_score_weighted = precision
    precision_score_macro = overall_metrics['precision']
    recall_score_weighted = recall
    recall_score_macro = overall_metrics['recall']
    
    # Extract per-class metrics
    F1n = np.array([per_class_metrics[cls]['f1_score'] for cls in row_labels])
    precision_score_none = np.array([per_class_metrics[cls]['precision'] for cls in row_labels])
    recall_score_none = np.array([per_class_metrics[cls]['recall'] for cls in row_labels])
    
    # Calculate fbeta scores (not directly available in calculate_results.py)
    # Using the formula: fbeta = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
    beta05 = 0.5
    beta2 = 2.0
    
    def calculate_fbeta(precision, recall, beta):
        return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)
    
    fbeta05_score_weighted = calculate_fbeta(precision_score_weighted, recall_score_weighted, beta05)
    fbeta05_score_macro = calculate_fbeta(precision_score_macro, recall_score_macro, beta05)
    fbeta05_score_none = np.array([calculate_fbeta(p, r, beta05) for p, r in zip(precision_score_none, recall_score_none)])
    
    fbeta2_score_weighted = calculate_fbeta(precision_score_weighted, recall_score_weighted, beta2)
    fbeta2_score_macro = calculate_fbeta(precision_score_macro, recall_score_macro, beta2)
    fbeta2_score_none = np.array([calculate_fbeta(p, r, beta2) for p, r in zip(precision_score_none, recall_score_none)])
    
    # Approximate accuracy (not really applicable to object detection but keeping for compatibility)
    acc = (precision_score_weighted + recall_score_weighted) / 2
    
    # Format as classification report-like string
    the_report = f"              precision    recall  f1-score\n"
    for i, cls in enumerate(row_labels):
        class_name = target_names[int(cls)] if int(cls) < len(target_names) else f"Class {cls}"
        the_report += f"{class_name:15} {per_class_metrics[cls]['precision']:.2f}    {per_class_metrics[cls]['recall']:.2f}    {per_class_metrics[cls]['f1_score']:.2f}\n"
    
    # Save confusion matrix plots if requested
    if save_png:
        plt.figure(figsize=(10,10))
        name = 'Normalise Confusion Matrix ' + title + ' Post bbox matching normalised'
        
        # Create dataframe for the confusion matrix
        df_confusion = pd.DataFrame(conf_mat, index=row_labels, columns=col_labels)
        
        # Map numeric class labels to names if needed
        mapped_row_labels = [target_names[int(label)] if int(label) < len(target_names) else f"Class {label}" for label in row_labels]
        mapped_col_labels = [(target_names[int(label)] if int(label) < len(target_names) else f"Class {label}") if label != "Non-detected" else label for label in col_labels]
        
        # Create normalized confusion matrix
        df_conf_norm = df_confusion.div(df_confusion.sum(axis=1), axis="index")
        plt.title(title)
        sns.heatmap(df_conf_norm, cmap='coolwarm', annot=True, annot_kws={"size": 16}, 
                   xticklabels=mapped_col_labels, yticklabels=mapped_row_labels)
        plt.savefig(os.path.join(path, name + '.png'), bbox_inches='tight')
        
        # Create regular confusion matrix
        plt.figure(figsize=(10,10))
        name = 'Confusion Matrix ' + title + ' Post bbox matching'
        plt.title(title)
        sns.heatmap(df_confusion, cmap='coolwarm', annot=True, annot_kws={"size": 16}, 
                   xticklabels=mapped_col_labels, yticklabels=mapped_row_labels)
        plt.savefig(os.path.join(path, name + '.png'), bbox_inches='tight')
        
        # Also save class distribution
        count_classes_file(gt_file, True, os.path.join(path, title + '_split.png'), target_names)
    
    # Save metrics to text file if requested
    if save_txt:
        with open(os.path.join(path, title + '.txt'), 'w') as file:
            file.write("F1 macro: " + str(F1m) + '\n')
            file.write("F1 weighted: " + str(F1w) + '\n')
            file.write("F1 none: " + str(F1n) + '\n')
            file.write("Accuracy score: " + str(acc) + '\n')
            file.write(the_report + '\n')
            file.write("Precision score weighted: " + str(precision_score_weighted) + '\n')
            file.write("Precision score macro: " + str(precision_score_macro) + '\n')
            file.write("Precision score none: " + str(precision_score_none) + '\n')
            file.write("Recall score weighted: " + str(recall_score_weighted) + '\n')
            file.write("Recall score macro: " + str(recall_score_macro) + '\n')
            file.write("Recall score none: " + str(recall_score_none) + '\n')
            file.write("Fbeta05 score weighted: " + str(fbeta05_score_weighted) + '\n')
            file.write("Fbeta05 score macro: " + str(fbeta05_score_macro) + '\n')
            file.write("Fbeta05 score none: " + str(fbeta05_score_none) + '\n')
            file.write("Fbeta2 score weighted: " + str(fbeta2_score_weighted) + '\n')
            file.write("Fbeta2 score macro: " + str(fbeta2_score_macro) + '\n')
            file.write("Fbeta2 score none: " + str(fbeta2_score_none) + '\n')
    
    # Return metrics as before
    return F1w, F1m, acc, precision_score_weighted, precision_score_macro, recall_score_weighted, recall_score_macro, fbeta05_score_weighted, fbeta05_score_macro, fbeta2_score_weighted, fbeta2_score_macro


