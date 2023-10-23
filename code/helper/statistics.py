import os 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from code.helper.utils import read_classes, match_classes

def get_distribution_plots(gt_file, save_name='Distribution_plots', save_path='/home/as-hunt/', obj_names='/home/as-hunt/Etra-Space/pha/obj.names'):
    classes = []
    bbox_areas = []
    with open(gt_file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            li = line.split(' ')
            classes.append(match_classes(int(li[1]), read_classes(obj_names)))
            bbox_areas.append(abs((int(li[4]) - int(li[2])) * (int(li[5]) - int(li[3]))))
    df = pd.DataFrame({'classes': classes, 'bbox_areas': bbox_areas})
    plt.figure(figsize=(10, 10))
    sns.set_theme(style='whitegrid')
    sns.displot(df, legend=True, x='bbox_areas', hue='classes', kind='kde')
    plt.xlabel('Bounding box area (in pixels)')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(save_path, save_name + '_combined'))
    plt.clf()
    sns.displot(df, legend=True, x='bbox_areas', hue='classes', log_scale=True, kind='kde')
    plt.xlabel('Bounding box area (in pixels log10)')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(save_path, save_name + '_combined_log'))
    plt.clf()
    unique_classes = list(set(classes))
    count = len(unique_classes)
    sns.displot(df, legend=True, x='bbox_areas', hue='classes', kind='kde', col='classes', col_wrap=count)
    plt.xlabel('Bounding box area (in pixels)')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(save_path, save_name + '_separate'))