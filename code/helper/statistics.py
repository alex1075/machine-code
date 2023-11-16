import os 
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
from code.helper.utils import read_classes, match_classes
from code.helper.annotations import make_ground_truth

def get_distribution_plots(gt_file, save_name='Distribution_plots', save_path='/home/as-hunt/', obj_names='/home/as-hunt/Etra-Space/pha/obj.names'):
    classes = []
    bbox_areas = []
    bbox_ln_areas = []
    with open(gt_file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            li = line.split(' ')
            classes.append(match_classes(int(li[1]), read_classes(obj_names)))
            bbox_areas.append(abs((int(li[4]) - int(li[2])) * (int(li[5]) - int(li[3]))))
            bbox_ln_areas.append(math.log(abs((int(li[4]) - int(li[2])) * (int(li[5]) - int(li[3])))))
    df = pd.DataFrame({'classes': classes, 'bbox_areas': bbox_areas})
    plt.figure(figsize=(10, 10))
    sns.set_theme(style='whitegrid')
    sns.displot(df, legend=True, x='bbox_areas', hue='classes', kind='kde', fill=True)
    plt.xlabel('Bounding box area (in pixels)')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(save_path, save_name + '_combined'))
    plt.clf()
    sns.kdeplot(df, legend=True, x='bbox_areas', hue='classes', log_scale=True, bw_method=0.1, fill=True)
    plt.xlabel('Bounding box area (in pixels smoothing 0.1')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(save_path, save_name + '_combined_smoothing0-1'))
    plt.clf()
    sns.kdeplot(df, legend=True, x='bbox_areas', hue='classes', log_scale=True, bw_method=0.2, fill=True)
    plt.xlabel('Bounding box area (in pixels smoothing 0.2')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(save_path, save_name + '_combined_smoothing0-2'))
    plt.clf()
    sns.kdeplot(df, legend=True, x='bbox_areas', hue='classes', log_scale=True, bw_method=0.5, fill=True)
    plt.xlabel('Bounding box area (in pixels smoothing 0.5')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(save_path, save_name + '_combined_smoothing0-5'))
    plt.clf()
    dn = pd.DataFrame({'classes': classes, 'bbox_ln_areas': bbox_ln_areas})
    sns.displot(dn, legend=True, x='bbox_ln_areas', hue='classes', kind='kde', fill=True)
    plt.xlabel('Bounding box area (in pixels ln)')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(save_path, save_name + '_combined_ln'))
    unique_classes = list(set(classes))
    count = len(unique_classes)
    sns.displot(df, legend=True, x='bbox_areas', hue='classes', kind='kde', col='classes', col_wrap=count, fill=True)
    plt.xlabel('Bounding box area (in pixels)')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(save_path, save_name + '_separate'))
    plt.clf()

def make_model_folder(path_to_folder, name, save_directory, obj_names):
    make_ground_truth(path_to_folder + 'gt.txt', path_to_folder)
    get_distribution_plots(path_to_folder + 'gt.txt', name, save_directory, obj_names)
