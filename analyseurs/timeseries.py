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
import pandas as pd
import tifffile
from bs4 import BeautifulSoup

from pyimfcs.class_imFCS import StackFCS

def time_str_2_min(creation_date):
    creation_time = creation_date.rstrip('\n')[-8:].split(':')
    crt_min = int(creation_time[0])*60+int(creation_time[1])+float(creation_time[2])/60
    return crt_min

def get_tiff_creationtime(file):
    
    with tifffile.TiffFile(file) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value
        image = tif.pages[0].asarray()
    metadata = tif_tags['IJMetadata']['Info'].split('\n')
    creationdate = list(filter(lambda x: "CreationDate" in x,metadata))
    assert(len(creationdate)==1)
    creationdate = creationdate[0].split('=')[-1].strip(' ')
    return time_str_2_min(creationdate)

def get_tiff_frameinterval_ms(file):
    with tifffile.TiffFile(file) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value
        image = tif.pages[0].asarray()
    metadata = tif_tags['IJMetadata']['Info'].split('\n')
    fint = list(filter(lambda x: "SetIntervalAction|Interval|TimeSpan|Value" in x,metadata))
    assert(len(fint)==1)
    fint = fint[0].split('=')[-1].strip(' ')
    unit = list(filter(lambda x: "SetIntervalAction|Interval|TimeSpan|DefaultUnitFormat" in x,metadata))
    unit = unit[0].split('=')[-1].strip(' ')
    assert unit=="ms"
    return float(fint)

def read_czi_metadata(file):
    """Reads creation date and frame interval (if applicable) from a czifile"""
    ff = czifile.CziFile(file)
    metadata = ff.metadata()
    bs = BeautifulSoup(metadata)
    creationdate = bs.information.document.creationdate.contents[0]
    
    finterval=0
    if bs.experiment.acquisitionblock.timeseries:
        try:
            print("is timeseries")
            frate  = bs.experiment.acquisitionblock.timeseriessetup.switch.switchaction.setintervalaction.interval.timespan
            finterval = float(frate.value.contents[0])
            frate_unit = frate.defaultunitformat.contents[0]
            assert frate_unit=='ms'
        except Exception as e:
            print(e)
    return time_str_2_min(creationdate), finterval

def extract_timeseries(files, out_time, nsum=3,
                       time_threshold = 700, intensity_threshold = 0.8,chi_threshold=0.015,
                       of_interest = "D [µm²/s]", use_mask=False, average_by_mask = False):
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
        # print('Acquisition time in mins:',acqtime_min)
        results = stack.extract_results(ith = intensity_threshold, 
                          chi_threshold = chi_threshold, use_mask=use_mask)
        
        values_of_interest = results[of_interest][nsum]
        if average_by_mask and use_mask:
            dff=pd.DataFrame(np.array(
                [results[of_interest][2],results['indices'][2]]).T,
                columns=[of_interest,"Label"])
            values_of_interest = dff.groupby("Label").mean()[of_interest].values
        metadata = stack.metadata_dict
        datekey = list(filter(lambda x: "Date" in x, metadata.keys()))
        assert len(datekey)==1
        creation_date = metadata[datekey[0]]
        try:
            creation_date=creation_date.decode('utf-8')
        except:
            print('Creation date could not be decoded')
            pass
        crt = time_str_2_min(creation_date)
        all_times.append(crt-out_time_min+acqtime_min/2)
        all_of_interest.append(values_of_interest)
        
    all_times = np.array(all_times)
    if out_time is None:
        all_times-=all_times.min()
    order = np.arange(len(all_times))
    all_of_interest = [y for x,y in sorted(zip(all_times,all_of_interest))]
    order = [y for x,y in sorted(zip(all_times,order))]
    all_times = sorted(all_times)
    return all_times, all_of_interest, order

def get_filenr(file):
    file_nr = file.split("Image ")[-1].split('_')[0].split('.')[0]

    return int(file_nr)

def sort_byfilenr(files):
    print(files)
    files_nrs = [get_filenr(f) for f in files]
    return [y for x,y in sorted(zip(files_nrs,files))]

def sort_byacqtime(files):
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
                    frac=0.2, savename=None,out_time = None, nsum=3,
                    intensity_threshold=0.8, chi_th=0.015,supp_folders=['before_time']):
    """Creates a visual summary of an FCS experiment, showing time-dependent results
    in a Matplotlib graph. This function will look for a file 'out_time.txt' describing
    experimental start time, in the form hh:mm:ss (e.g 13:24:00)
    Parameters:
        path (str): main path where to find the data
        ofinterest (str or list): list of quantities to plot vs time on slide. 
            These are keys to result dictionary from the method extract_results
            of StackFCS class.
        zoom (int): zoom factor for BF images
        frac (float): between 0 and 1, saturation factor for BF images
        savename (str): if provided, saves the plot at this name
        out_time (str): if 'out_time.txt' is not found. Optional
        nsum (int): binning
        intensity_threshold (float): as usual
        chi_th (float): as usual
        supp_folders (list): optional, if provided adds results from supplementary folders
        """
    if type(of_interest)==str:
        of_interest = [of_interest]
    files = glob.glob(path+"*.h5")
    next_images = [get_next_file(w,path) for w in files]

    if supp_folders is not None:
        for sf in supp_folders:
            supfiles=glob.glob(path+sf+"/*.h5")
            files.extend(supfiles)
            next_images.extend( [get_next_file(w,path+sf+"/") for w in supfiles])
    files = sort_byfilenr(files)
    next_images = sort_byfilenr(next_images)
    try:
        with open(path+"out_time.txt","r") as f:
            aa= f.read()
        out_time=aa[:8]
    except:
        pass
    fig = plt.figure(figsize=(10,4))
    # plots the different quantities
    for j,oi in enumerate(of_interest):
        times, vals, order = extract_timeseries(files,out_time,
                                         of_interest=oi,nsum=nsum,use_mask=True,
                                         intensity_threshold=intensity_threshold,
                                         chi_threshold=chi_th)
        
        vmeans = [np.median(w) for w in vals]
        vstd = [np.std(w) for w in vals]
        
        ax1 = fig.add_subplot(2,len(of_interest),len(of_interest)+j+1)
        ax1.errorbar(times,vmeans,yerr=vstd,marker="o",capsize=5)
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel(oi)
    # plots images
    for j in range(len(next_images)):
        next_filename = next_images[order[j]]
        ax0 = fig.add_subplot(2,len(next_images),j+1)
        try:
            ax0.set_title(os.path.split(next_filename)[-1].rstrip('.czi'))
            img = czifile.imread(next_filename).squeeze()
            print(img.shape)
            u,v=img.shape
            nnu = (u-u//zoom)//2
            nnv = (v-v//zoom)//2
            if zoom>1:
                img = img[nnu:-nnu,nnv:-nnv]
            ampl = img.max()-img.min()
            ax0.imshow(img,cmap="gray",vmin=img.min()+ampl*frac,
                       vmax=img.max()-ampl*frac)
            ax0.axis('off')
        except Exception as e:
            print(e)
            pass
    
    # fig.tight_layout()
    if savename is not None:
        fig.savefig(savename+"_"+".png")

def FCS_timelapse(files, of_interest = ['diffusion_coefficients'], 
                    savename=None,path_out_time = None, nsum=2, 
                    axes = None, label = None):
    """Extracts the results of an experiment over time"""

    files = sort_byfilenr(files)
    try:
        with open(path_out_time+"out_time.txt","r") as f:
            aa= f.read()
        out_time=aa[:8]
    except:
        pass
    out_vals={}
    if axes is None:
        fig = plt.figure(figsize=(5,3))
    # plots the different quantities
    for j,oi in enumerate(of_interest):
        times, vals, order = extract_timeseries(files,out_time,
                                         of_interest=oi,nsum=nsum)
        
        vmeans = [np.median(w) for w in vals]
        vstd = [np.std(w) for w in vals]
        if axes is None:
            ax1 = fig.add_subplot(1,len(of_interest),j+1)
        else:
            ax1=axes[j]
        ax1.errorbar(times,vmeans,yerr=vstd,marker="o",capsize=5,label=label)
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel(oi)
        out_vals[oi]=[times,vmeans,vstd]
    # fig.tight_layout()
    if savename is not None:
        fig.savefig(savename+"_"+".png")
    return out_vals

def pool(l1, lref, fcondition):
    out = []
    nr=0
    for elt, tt in zip(l1,lref):
        if fcondition(tt):
            out.extend(elt)
            nr+=1
    if nr<2:
        print('Warning, only ',nr,' elements found')
    return out
   
def tl_multiple_exp(folders, labels = None,colors = None,
                    of_interest="diffusion_coefficients", plot_violins=True,
                    time_threshold=700, intervalspace = 2):
    if labels is None:
        labels = [w.split('/')[-2] for w in folders]

    fig,axes = plt.subplots(1,2,figsize=(8,4))
    ax=axes[0]
    all_times = list()
    all_vals = list()
    for j, path in enumerate(folders):
        files = glob.glob(path+"*.h5")
        
        # files = sort_byfilenr(files)
        try:
            with open(path+"out_time.txt","r") as f:
                aa= f.read()
            out_time=aa[:8]
        except:
            out_time=None
        times, vals, order = extract_timeseries(files,out_time,
                                         of_interest=of_interest)
        all_times.extend(times)
        all_vals.extend(vals)
        vmeans = [np.mean(w) for w in vals]
        vstd = [np.std(w) for w in vals]
        ax.errorbar(times,vmeans,yerr=vstd,marker="o",capsize=5,label=labels[j])
    ax.legend()
    ax.set_xlabel('Time (min)')
    ax.set_ylabel(of_interest)
    fig.tight_layout()
    
    if plot_violins:
        intervals = np.arange(0,time_threshold,intervalspace)
        times = np.asarray(all_times)
        final_diffs = list()
        to_remove =[]
        for j in range(len(intervals)-1):
            fcond = lambda x: x>intervals[j] and x<= intervals[j+1]
            oo = pool(all_vals,times,fcond)
            if len(oo)==0:
                to_remove.append(j)
            else:
                final_diffs.append(oo)
        
        fcond = lambda x: x>intervals[-1]
        
        oo = pool(all_vals,times,fcond)
        if len(oo)==0:
            to_remove.append(j+1)
        else:
            final_diffs.append(oo)
        # print(final_diffs)
        msk = np.arange(len(intervals))
        msk = np.array([w not in to_remove for w in msk])
        intervals = intervals[msk]
        intervals+=intervalspace
        wdth=1.5
        ax2 =axes[1]
        ax2.axhline(np.median([np.median(w) for w in final_diffs]),color="k",linestyle='--')
        ax2.violinplot(final_diffs,positions=intervals,widths=wdth,showextrema=False) 
        ax2.boxplot(final_diffs,positions=intervals,widths=wdth)
        ax2.plot(intervals,[np.median(w) for w in final_diffs],marker='o', color='k')
        # msk_scatter = times<time_threshold
        # plt.scatter(all_times_pooled[msk_scatter],all_diffs_pooled[msk_scatter])
        ax2.set_ylim(bottom=0)
        ax2.set_xlabel('Time on microscope (min)')
        ax2.set_ylabel(of_interest)