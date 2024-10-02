#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:22:30 2023

@author: aurelienb
"""
from os import listdir
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

def read_biofilm_volume_results(path: str, get_ratio1to2 = False):
    filenames = find_csv_filenames(path)

    global_filenames = []
    for name in filenames:
        if name[-10:-4] == 'global':
            global_filenames.append(name)
    result_name_1 = "Intensity_Mean_ch1_mean_biovolume"
    result_name_alternative = "Intensity_Mean_ch2_mean_biovolume"
    
    output_names = ["Intensity_Mean_ch1_mean_biovolume",
                   "Intensity_Mean_ch1_mean"]
    default_value="ch1"
    replace_value="ch2"
    output_names2 = ["Intensity_Ratio_Mean_ch2_ch1_noBackground_mean_biovolume"]
    default_value2="ch2_ch1"
    replace_value2="ch1_ch2"
    
    if get_ratio1to2:
        cols=["Biofilm_Volume"]+output_names+\
                               [w.replace(default_value,replace_value) for w in output_names]+output_names2+\
                               [w.replace(default_value2,replace_value2) for w in output_names2]
    else:
        cols=["Biofilm_Volume"]+output_names+\
                               [w.replace(default_value,replace_value) for w in output_names]
    results = pd.DataFrame(index=global_filenames, columns=cols)

    for name in filenames:
        if name[-10:-4] == 'global':
            df_temp = pd.read_csv(path + name)
            results.loc[name, "Biofilm_Volume"] = df_temp.loc[1,"Biofilm_Volume"]
            for on in output_names:
                try:
                    results.loc[name, on] = df_temp.loc[1,on]
                except:
                    results.loc[name, on.replace(default_value,replace_value)] = df_temp.loc[1,on.replace(default_value,replace_value)]
            if get_ratio1to2:
                for on in output_names2:
                    try:
                        results.loc[name, on] = df_temp.loc[1,on]
                    except:
                        results.loc[name, on.replace(default_value2,replace_value2)] = df_temp.loc[1,on.replace(default_value2,replace_value2)]

    results = results.astype(float)
    return results


def read_biofilm_volume_results_new(path: str, get_ratio1to2 = False):
    filenames = find_csv_filenames(path)

    global_filenames = []
    for name in filenames:
        if name[-10:-4] == 'global':
            global_filenames.append(name)
    result_name_1 = "Intensity_Mean_ch1_mean_biovolume"
    result_name_alternative = "Intensity_Mean_ch2_mean_biovolume"
    
    output_names = ["Intensity_Mean_ch1_mean_biovolume",
                   "Intensity_Mean_ch1_mean"]
    default_value="ch1"
    replace_value="ch2"
    output_names2 = ["Intensity_Ratio_Mean_ch2_ch1_noBackground_mean_biovolume"]
    default_value2="ch2_ch1"
    replace_value2="ch1_ch2"
    
    if get_ratio1to2:
        cols=["Biofilm_Volume"]+output_names+\
                               [w.replace(default_value,replace_value) for w in output_names]+output_names2+\
                               [w.replace(default_value2,replace_value2) for w in output_names2]
    else:
        cols=["Biofilm_Volume"]+output_names+\
                               [w.replace(default_value,replace_value) for w in output_names]
    results = pd.DataFrame(index=global_filenames, columns=cols)

    for name in global_filenames:
        df_temp = pd.read_csv(path + name)
        results.loc[name, "Biofilm_Volume"] = df_temp.loc[1,"Biofilm_Volume"]
        for on in output_names:
            try:
                results.loc[name, on] = df_temp.loc[1,on]
            except:
                results.loc[name, on.replace(default_value,replace_value)] = df_temp.loc[1,on.replace(default_value,replace_value)]
        if get_ratio1to2:
            for on in output_names2:
                try:
                    results.loc[name, on] = df_temp.loc[1,on]
                except:
                    results.loc[name, on.replace(default_value2,replace_value2)] = df_temp.loc[1,on.replace(default_value2,replace_value2)]

    results = results.astype(float)
    return results

def extract_ratios(results, savepath, quantity='Biofilm_Volume',new_name=None):
    """Extracts ratios of quantity from a dataframe at different positions over time, and plots the results.
    Parameters:
    results (Dataframe): has to contain a value for given quantity in each cell
    savepath (str): where to save the corresponding curves
    quantity (str): key of quantity to extract from Dataframe"""
    indices = list(results[quantity].index)
    values = np.array(list(results['Biofilm_Volume'].values),dtype=float)

    frame_nrs = np.array([int(name.split('_frame')[-1].split('_global.csv')[0]) for name in indices])
    # channels
    channels = np.array([int(name.split('_frame')[0].split('_ch')[-1]) for name in indices])
    nchannels = np.unique(channels)

    if len(nchannels)==1:
        raise ValueError('Only one channel detected in this dataset')
    elif len(nchannels)==2:
        print('2 channels detected')
    else:
        print('Warning!!! {} channels detected'.format(len(nchannels)))

    # positions
    positions = np.array([int(name.split('_pos')[-1].split('_ch')[0]) for name in indices])
    npos = np.unique(positions)
    if len(npos)<2:
        print('Warning!! Only {} positions found'.format(len(npos)))
    else:
        print("found {} positions".format(len(npos)))

    ratios = []
    output_per_channel = dict(zip(nchannels,[[] for w in nchannels]))
    for pos in npos:
        sorted_outputs = []
        for ch in nchannels:
            fns = frame_nrs[np.logical_and(channels==ch,positions==pos)]
            vals = values[np.logical_and(channels==ch,positions==pos)]
            out = np.array(sorted(zip(fns,vals)))
            sorted_outputs.append(out)
            output_per_channel[ch].append(out)
         
        assert len(sorted_outputs)==2
        assert (sorted_outputs[0][:,0]==sorted_outputs[1][:,0]).all()
        frame_numbers = sorted_outputs[0][:,0]
        ratio = sorted_outputs[1][:,1]/sorted_outputs[0][:,1]
        ratios.append(np.array([frame_numbers,ratio]))
                   
    # Plotting
    fig,axes = plt.subplots(1,3,figsize=(10,3))
    fig.suptitle(new_name)
    for j,rts in enumerate(ratios):
        axes[0].plot(rts[0],rts[1],label='pos{}'.format(npos[j]))
    axes[0].legend(bbox_to_anchor=(1.1, 1.05))
    
    ratios = np.array(ratios)
    if len(npos)>1:
        mean_ratio = ratios.mean(axis=0)
        std_ratio = ratios.std(axis=0)
        axes[1].errorbar(mean_ratio[0],mean_ratio[1],yerr=std_ratio[1],capsize=5)
    # Plot error of individual channel curves
    colors={1:'g',2:'m'}
    for ch in output_per_channel.keys():
        volumes_channel = np.asarray(output_per_channel[ch])
        vmean = volumes_channel.mean(axis=0)
        vstd = volumes_channel.std(axis=0)
        axes[2].errorbar(vmean[:,0],vmean[:,1], yerr=vstd[:,1],capsize=5, 
                         label='Channel {}'.format(ch),color=colors[ch])                      
    axes[0].set_xlabel("# Frame")
    axes[0].set_ylabel("Ratio ch{}/ch{}".format(nchannels[1],nchannels[0]))
    axes[1].set_xlabel("# Frame")
    axes[1].set_ylabel("Ratio ch{}/ch{}".format(nchannels[1],nchannels[0]))
    axes[2].set_xlabel("Frame Numbers")
    axes[2].set_ylabel("Biofilm volume")
    axes[2].legend(bbox_to_anchor=(1.1, 1.05))
                              
    axes[0].set_title('All positions')
    axes[1].set_title('Mean+/-std across positions')
    axes[2].set_title('Biofilm volume per channel')
    # axes[2].legend()
    fig.tight_layout()
    fig.savefig(savepath,facecolor='white',dpi=600)

plt.close('all')

path = "/home/aurelienb/Desktop/Data_plus_last script/debug/50 Gm + 32 COL + 24h 8 IVA/"

df1 = read_biofilm_volume_results(path)
df2 = read_biofilm_volume_results_new(path)
new_name=os.path.split(os.path.dirname(path))[-1]
# extract_ratios(df1,"test.png",new_name=new_name)

results = df1
savepath="test.png"
indices = list(results['Biofilm_Volume'].index)
values = np.array(list(results['Biofilm_Volume'].values),dtype=float)

frame_nrs = np.array([int(name.split('_frame')[-1].split('_global.csv')[0]) for name in indices])
# channels
channels = np.array([int(name.split('_frame')[0].split('_ch')[-1]) for name in indices])
nchannels = np.unique(channels)

if len(nchannels)==1:
    raise ValueError('Only one channel detected in this dataset')
elif len(nchannels)==2:
    print('2 channels detected')
else:
    print('Warning!!! {} channels detected'.format(len(nchannels)))

# positions
positions = np.array([int(name.split('_pos')[-1].split('_ch')[0]) for name in indices])
npos = np.unique(positions)
if len(npos)<2:
    print('Warning!! Only {} positions found'.format(len(npos)))
else:
    print("found {} positions".format(len(npos)))

ratios = []
output_per_channel = dict(zip(nchannels,[[] for w in nchannels]))
for pos in npos:
    sorted_outputs = []
    for ch in nchannels:
        fns = frame_nrs[np.logical_and(channels==ch,positions==pos)]
        vals = values[np.logical_and(channels==ch,positions==pos)]
        out = np.array(sorted(zip(fns,vals)))
        sorted_outputs.append(out)
        output_per_channel[ch].append(out)
     
    assert len(sorted_outputs)==2
    assert (sorted_outputs[0][:,0]==sorted_outputs[1][:,0]).all()
    frame_numbers = sorted_outputs[0][:,0]
    ratio = sorted_outputs[1][:,1]/sorted_outputs[0][:,1]
    ratios.append(np.array([frame_numbers,ratio]))
               
# Plotting
fig,axes = plt.subplots(1,3,figsize=(10,3))
fig.suptitle(new_name)
for j,rts in enumerate(ratios):
    axes[0].plot(rts[0],rts[1],label='pos{}'.format(npos[j]))
axes[0].legend(bbox_to_anchor=(1.1, 1.05))

ratios = np.array(ratios)
if len(npos)>1:
    mean_ratio = ratios.mean(axis=0)
    std_ratio = ratios.std(axis=0)
    axes[1].errorbar(mean_ratio[0],mean_ratio[1],yerr=std_ratio[1],capsize=5)
# Plot error of individual channel curves
colors={1:'g',2:'m'}
for ch in output_per_channel.keys():
    volumes_channel = np.asarray(output_per_channel[ch])
    vmean = volumes_channel.mean(axis=0)
    vstd = volumes_channel.std(axis=0)
    axes[2].errorbar(vmean[:,0],vmean[:,1], yerr=vstd[:,1],capsize=5, 
                     label='Channel {}'.format(ch),color=colors[ch])                      
axes[0].set_xlabel("# Frame")
axes[0].set_ylabel("Ratio ch{}/ch{}".format(nchannels[1],nchannels[0]))
axes[1].set_xlabel("# Frame")
axes[1].set_ylabel("Ratio ch{}/ch{}".format(nchannels[1],nchannels[0]))
axes[2].set_xlabel("Frame Numbers")
axes[2].set_ylabel("Biofilm volume")
axes[2].legend(bbox_to_anchor=(1.1, 1.05))
                          
axes[0].set_title('All positions')
axes[1].set_title('Mean+/-std across positions')
axes[2].set_title('Biofilm volume per channel')
# axes[2].legend()
fig.tight_layout()
fig.savefig(savepath,facecolor='white',dpi=600)
