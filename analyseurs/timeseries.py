#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:34:39 2023

@author: aurelienb
"""
import glob
import os
import czifile
import numpy as np
import matplotlib.pyplot as plt

from pyimfcs.class_imFCS import StackFCS

def time_str_2_min(creation_date):
    creation_time = creation_date[-8:].split(':')
    crt_min = int(creation_time[0])*60+int(creation_time[1])+float(creation_time[2])/60
    return crt_min

def extract_timeseries(files, out_time, nsum=3,
                       time_threshold = 700, intensity_threshold = 0.8,chi_threshold=0.015,
                       of_interest = "diffusion_coefficients", use_mask=False):
    """Extracts all parameter of interest from at different times"""
    if out_time is not None:
        out_time_min = time_str_2_min(out_time)
    else:
        out_time_min = 0
        
    all_of_interest = []
    all_times = []
    for file in files:
        stack = StackFCS(file,load_stack=False)
        stack.load(light_version=True)
        acqtime_min = stack.fcs_curves_dict[nsum][0,0][-1,0]*1.5/60
        results = stack.extract_results(ith = intensity_threshold, 
                          chi_threshold = chi_threshold, use_mask=use_mask)
        
        values_of_interest = results[of_interest][nsum]
        metadata = stack.metadata_dict
        datekey = list(filter(lambda x: "Date" in x, metadata.keys()))
        assert len(datekey)==1
        creation_date = metadata[datekey[0]]
        try:
            creation_date=creation_date.decode('utf-8')
        except:
            pass
        crt = time_str_2_min(creation_date)
        all_times.append(crt-out_time_min+acqtime_min/2)
        all_of_interest.append(values_of_interest)
        
    all_times = np.array(all_times)
    all_of_interest = [y for x,y in sorted(zip(all_times,all_of_interest))]
    all_times = sorted(all_times)
    return all_times, all_of_interest

def get_filenr(file):
    file_nr = file.split("Image ")[-1].split('_')[0].split('.')[0]
    return int(file_nr)

def sort_byfilenr(files):
    print(files)
    files_nrs = [get_filenr(f) for f in files]
    return [y for x,y in sorted(zip(files_nrs,files))]

def get_next_file(file, path):
    """Given an image number, finds the image with next number"""
    all_files = glob.glob(path+"*.czi")
    files_nrs = np.array([get_filenr(f) for f in all_files])
    nr = get_filenr(file)
    tb = files_nrs - nr
    indices = np.where(tb>0)[0]
    if len(indices)==0:
        return -1
    tb = tb[indices]
    i0 = np.where(tb==tb.min())[0][0]

    index = indices[i0]
    return all_files[index]

def summarise_slide(path, of_interest = 'diffusion_coefficients', zoom=3, 
                    frac=0.2, savename=None,out_time = None):
    if type(of_interest)==str:
        pass
    files = glob.glob(path+"*.h5")
    files = sort_byfilenr(files)
    try:
        with open(path+"out_time.txt","r") as f:
            aa= f.read()
        out_time=aa[:8]
    except:
        pass
    next_images = [get_next_file(w,path) for w in files]
  
    fig = plt.figure(figsize=(10,4))
    for j in range(len(next_images)):
        next_filename = next_images[j]
        ax0 = fig.add_subplot(2,len(next_images),j+1)
        try:
            ax0.set_title(os.path.split(next_filename)[-1])
            img = czifile.imread(next_filename).squeeze()
            u,v=img.shape
            nnu = (u-u//zoom)//2
            nnv = (v-v//zoom)//2
            img = img[nnu:-nnu,nnv:-nnv]
            ampl = img.max()-img.min()
            ax0.imshow(img,cmap="gray",vmin=img.min()+ampl*frac,
                       vmax=img.max()-ampl*frac)
        except Exception as e:
            print(e)
            pass
    
    times, vals = extract_timeseries(files,out_time,
                                     of_interest=of_interest)
    
    vmeans = [np.mean(w) for w in vals]
    vstd = [np.std(w) for w in vals]
    
    ax1 = fig.add_subplot(2,1,2)
    ax1.errorbar(times,vmeans,yerr=vstd,marker="o",capsize=5)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel(of_interest)
    fig.tight_layout()
    if savename is not None:
        fig.savefig(savename+"_"+of_interest+".png")
    
def tl_multiple_exp(folders, labels = None,colors = None,
                    of_interest="diffusion_coefficients"):
    if labels is None:
        labels = [w.split('/')[-2] for w in folders]
    
    fig,ax = plt.subplots(1,1)
    for j, path in enumerate(folders):
        files = glob.glob(path+"*.h5")
        
        files = sort_byfilenr(files)
        try:
            with open(path+"out_time.txt","r") as f:
                aa= f.read()
            out_time=aa[:8]
        except:
            pass
        times, vals = extract_timeseries(files,out_time,
                                         of_interest=of_interest)
        
        vmeans = [np.mean(w) for w in vals]
        vstd = [np.std(w) for w in vals]
        ax.errorbar(times,vmeans,yerr=vstd,marker="o",capsize=5,label=labels[j])
    ax.legend()
    ax.set_xlabel('Time (min)')
    ax.set_ylabel(of_interest)
    fig.tight_layout()