import pandas as pd
import seaborn as sns
import numpy as np
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, fbeta_score
import matplotlib.pyplot as plt
from code.helper.annotations import *

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

def plot_bbox_area(gt_file, pd_file, save_name='areas', path='/home/as-hunt/'):
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
                        combined.append([(abs(int(bbax[2]) - int(bbax[0])) * abs(int(bbax[3]) - int(bbax[1]))), (abs(int(bbox[2]) - int(bbox[0])) * abs(int(bbox[3]) - int(bbox[1]))), float(classes), float(clisses), match, float(iou(bbax, bbox))])
                    else:   
                        match = False
                        combined.append([(abs(int(bbax[2]) - int(bbax[0])) * abs(int(bbax[3]) - int(bbax[1]))), (abs(int(bbox[2]) - int(bbox[0])) * abs(int(bbox[3]) - int(bbox[1]))), float(classes), float(clisses), match, float(iou(bbax, bbox))])
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


    fig.savefig(path + 'test2png.png', dpi=100)
    
    df = pd.DataFrame({'Class':classesp, 'Area':dfp, 'Dataset':tagp}, columns=["Class", "Area", "Dataset"])
    for i in range(len(classesg)):
        new_row = {'Class': classesg[i], 'Area': dfg[i], 'Dataset': tagg[i]}
        df = df.append(new_row, ignore_index=True)
    sns.violinplot(data=df, cut=0, x='Class', y='Area', inner='box', scale='count', hue="Dataset", split=True, ax=axs[0, 0])
    axs[0, 0].set_title('Bbox Area Plotting per Class')

    # plt.savefig(save_name+'_2.png', bbox_inches='tight')
    # plt.clf()
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
    x = du['x']
    y = du['y']
    z = du['IoU']
    w = du['PD_class']
    plt.rcParams["figure.figsize"] = [16, 10]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(x, y, z, c=w ,cmap='viridis')
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    ax.view_init(40, 60)
    plt.savefig(path + save_name +'_3D.png', bbox_inches='tight')        
    
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

def do_math(gt_file, pd_file, title, path, save_txt=False, obj_name='/home/as-hunt/Etra-Space/white-thirds/obj.names', save_png=False):
    '''This function takes in a ground truth file and a prediction file and returns AI metrics for the model
    Optionally, it also outputs a confusion matrix and a text file with the results of the confusion matrix
    
    _____________________________________________________________
    Args:

    gt_file: ground truth file
    pd_file: prediction file
    
    title: title of the output files
    path: path to save the output files
    
    save_txt: boolean, whether or not to save the text file
    save_png: boolean, whether or not to save the png file
    
    obj_name: path to the obj.names file to use for the confusion matrix
    ______________________________________________________________
    '''
    gt = open(gt_file, 'r')
    # print('tick')
    gt_array = []
    gt_len = 0
    gt_cm = []
    pud = open(pd_file, 'r')
    # print('tack')
    pd_len = 0
    pd_array = []
    pd_cm = []
    target_names = []
    temp = []
    # print(obj_name)
    with open(obj_name, 'r') as f:
        lines = (line.rstrip() for line in f)
        lines = list(line for line in lines if line) # Non-blank lines in a list
        for line in lines:
            # print(line)
            temp.append(line)      
    for item in temp:
        if item == 'ECHY':
            target_names.append('Echinocyte')
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
    target_names.sort()        
    # print('tock')
    for line in pud:
        li = line.split(' ')
        name = li[0]
        classes = li[1]
        bbox = [int(li[2]), int(li[3]), int(li[4]), int(li[5])]
        confidence = li[6]
        pd_array.append([name, bbox, classes, confidence])
        pd_len += 1
    for lune in gt:
        lu = lune.split(' ')
        nome = lu[0]
        clisses = lu[1]
        bbax = [int(lu[2]), int(lu[3]), int(lu[4]), int(lu[5])]
        gt_array.append([nome, bbax, clisses])
        gt_len += 1
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
                if iou(bbox, bbax) >= 0.5:
                        gt_cm.append(clisses)
                        pd_cm.append(classes)
                        gt_array.pop(place)
                else:
                    pass
    y_actu = pd.Series(gt_cm, name='Ground Truth')
    y_pred = pd.Series(pd_cm, name='Predicted')
    # print('tick')
    try:
        F1m = f1_score(y_actu, y_pred, average='macro')
        if math.isnan(F1m)==True:
            F1m =  '0'
        elif F1m  == '-0.0':
            F1m =  '0'    
    except:
        F1m =  '0'
    try:
        F1w = f1_score(y_actu, y_pred, average='weighted')
        if math.isnan(F1w)==True:
            F1w =  '0'
        elif F1w == '0.0':        
            F1w =  '0'
    except:
        F1w =  '0'
    F1n = f1_score(y_actu, y_pred, average=None)
    try:
        acc = accuracy_score(y_actu, y_pred)
        if math.isnan(acc)==True:
            acc =  '0'
        elif acc  == '-0.0':
            acc =  '0'
    except:
        acc =  '0'
    try:
        the_report = classification_report(y_actu, y_pred, target_names=target_names)
    except:
        the_report = 'Classification report failed'
    try:
        precision_score_weighted = precision_score(y_actu, y_pred, average='weighted')
        if math.isnan(precision_score_weighted)==True:
            precision_score_weighted =  '0'
        elif precision_score_weighted  == '-0.0':
            precision_score_weighted =  '0'    
    except:
        precision_score_weighted =  '0'
    try:
        precision_score_macro = precision_score(y_actu, y_pred, average='macro')
        if math.isnan(precision_score_macro)==True:
            precision_score_macro =  '0'
        elif precision_score_macro  == '-0.0':
            precision_score_macro =  '0'
    except:
        precision_score_macro =  '0'
    precision_score_none = precision_score(y_actu, y_pred, average=None)
    try:
        recall_score_weighted = recall_score(y_actu, y_pred, average='weighted')
        if math.isnan(recall_score_weighted)==True:
            recall_score_weighted =  '0'
        elif recall_score_weighted  == '-0.0':
            recall_score_weighted =  '0'
    except:
        recall_score_weighted =  '0'
    try:
        recall_score_macro = recall_score(y_actu, y_pred, average='macro')
        if math.isnan(recall_score_macro)==True:
            recall_score_macro =  '0'
        elif recall_score_macro  == '-0.0': 
            recall_score_macro =  '0'
    except:
        recall_score_macro =  '0'
    recall_score_none = recall_score(y_actu, y_pred, average=None)
    try:
        fbeta05_score_weighted = fbeta_score(y_actu, y_pred, average='weighted', beta=0.5)
        if math.isnan(fbeta05_score_weighted)==True:
            fbeta05_score_weighted =  '0'
        elif fbeta05_score_weighted  == '-0.0':
            fbeta05_score_weighted =  '0'
    except:
        fbeta05_score_weighted =  '0'
    try:
        fbeta05_score_macro = fbeta_score(y_actu, y_pred, average='macro', beta=0.5)
        if math.isnan(fbeta05_score_macro)==True:
            fbeta05_score_macro =  '0'
        elif fbeta05_score_macro  == '-0.0':
            fbeta05_score_macro =  '0'
    except:
        fbeta05_score_macro =  '0'
    fbeta05_score_none = fbeta_score(y_actu, y_pred, average=None, beta=0.5)
    try:
        fbeta2_score_weighted = fbeta_score(y_actu, y_pred, average='weighted', beta=2)
        if math.isnan(fbeta2_score_weighted)==True:
            fbeta2_score_weighted =  '0'
        elif fbeta2_score_weighted  == '-0.0':
            fbeta2_score_weighted =  '0'
    except:
        fbeta2_score_weighted =  '0'
    try:
        fbeta2_score_macro = fbeta_score(y_actu, y_pred, average='macro', beta=2)
        if math.isnan(fbeta2_score_macro)==True:
            fbeta2_score_macro =  '0'
        elif fbeta2_score_macro  == '-0.0':
            fbeta2_score_macro =  '0'
    except:
        fbeta2_score_macro =  '0'
    fbeta2_score_none = fbeta_score(y_actu, y_pred, average=None, beta=2)
    # print('tock')
    try:
        name = 'Normalise Confusion Matrix ' + title + ' Post bbox matching normalised'
        df_confusion = pd.crosstab(y_actu, y_pred, dropna=False)
        df_conf_norm = df_confusion.div(df_confusion.sum(axis=1), axis="index")
        plt.title(title)
        sns.set(font_scale=1.4) # for label size
        sns.heatmap(df_conf_norm, cmap='coolwarm', annot=True, annot_kws={"size": 16}, xticklabels=target_names, yticklabels=target_names) # font size
        tick_marks = np.arange(len(df_conf_norm.columns))
        if save_png == True:
            plt.savefig(path + name + '.png', bbox_inches='tight')
        plt.clf()
        name = 'Confusion Matrix ' + title + ' Post bbox matching'
        plt.title(title)
        sns.set(font_scale=1.4) # for label size
        sns.heatmap(df_confusion, cmap='coolwarm', annot=True, annot_kws={"size": 16}, xticklabels=target_names, yticklabels=target_names) # font size
        tick_marks = np.arange(len(df_confusion.columns))
        if save_png == True:
            plt.savefig(path + name + '.png', bbox_inches='tight')
            count_classes_file(gt_file, True, title + '_split.png', target_names)
    except:
        pass
    # print('tick')
    if save_txt == True:
        file = open(path + title + '.txt', 'w')
        file.write("F1 macro: " + str(F1m) + '\n')
        file.write("F1 weighted: " + str(F1w) + '\n')
        file.write("F1 none: " + str(F1n) + '\n')
        file.write("Accuracy score sklearn: " + str(acc) + '\n')
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
        file.close()
    # print('tock')    
    return F1w, F1m, acc, precision_score_weighted, precision_score_macro, recall_score_weighted, recall_score_macro, fbeta05_score_weighted, fbeta05_score_macro, fbeta2_score_weighted, fbeta2_score_macro

def inference_report(repot_txt, save_name='areas.png'):
    '''Plots the areas of the bounding boxes in the ground truth and prediction from txt summary files'''
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
