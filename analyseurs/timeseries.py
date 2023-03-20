#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:34:39 2023

@author: aurelienb
"""
import glob
import numpy as np

from pyimfcs.class_imFCS import StackFCS
def time_str_2_min(creation_date):
    creation_time = creation_date[-8:].split(':')
    crt_min = int(creation_time[0])*60+int(creation_time[1])+float(creation_time[2])/60
    return crt_min

def extract_timeseries_raw(path, out_time, acqtime_min = 4.5, nsum=3,
                       time_threshold = 700, intensity_threshold = 0.8,chi_threshold=0.015,
                       of_interest = "diffusion coefficients", use_mask=False):
    """Extracts all diffusion coefficients at different times"""
    out_time_min = time_str_2_min(out_time)
    files = glob.glob(path+"*.h5")
    print(files)
    
    all_of_interest = []
    all_times = []
    for file in files:
        stack = StackFCS(file,load_stack=False)
        stack.load(light_version=True)
        results = stack.extract_results(ith = intensity_threshold, 
                          chi_threshold = chi_threshold, use_mask=use_mask)
        values_of_interest = results[of_interest]
        metadata = stack.metadata
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
 
    return all_times, all_of_interest
