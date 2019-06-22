#!/usr/bin/env python
# coding: utf-8

# Required imports
import datetime
import os
import numpy as np
import pandas as pd
import gzip
import glob
import pickle
import copy
import math
from io import StringIO
import importlib.machinery
from scipy import stats, optimize
import multiprocessing
from functools import partial
import time

import scipy.signal as sig # Find peaks
# Anomaly detection
import banpei
import csv
# Changepoint detection
import ruptures as rpt
import bayesian_changepoint as bcp
import bayesian_changepoint_detection.bayesian_changepoint_detection.offline_changepoint_detection as offcd
import bayesian_changepoint_detection.bayesian_changepoint_detection.online_changepoint_detection as oncd


# In[ ]:
# Dataset parsers for header/ body for CSVs
def parse_header_of_csv(csv_str):
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index('\n')];
    columns = headline.split(',');

    # The first column should be timestamp:
    assert columns[0] == 'timestamp';
    # The last column should be label_source:
    assert columns[-1] == 'label_source';

    # Search for the column of the first label:
    for (ci,col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci;
            break;
        pass;

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind];
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1];
    for (li,label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('label:');
        label_names[li] = label.replace('label:','');
        pass;

    return (feature_names,label_names);

def parse_body_of_csv(csv_str,n_features):
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(StringIO(csv_str),delimiter=',',skiprows=1);

    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:,0].astype(int);

    # Read the sensor features:
    X = full_table[:,1:(n_features+1)];

    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:,(n_features+1):-1]; # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat); # M is the missing label matrix

    #print("M matrix shape:",M.shape)
    #print("Matrix: ",np.argwhere(M))
    trinary_labels_mat[M]=-1 # Replace NaNs with -1.0 for which we then apply a mask
    unique,counts=np.unique(trinary_labels_mat,return_counts=True)
    print(*zip(unique,counts))

#     Y = np.where(M,0,trinary_labels_mat) > 0.; # Y is the label matrix

    return (X,trinary_labels_mat,M,timestamps);

def read_user_data(directory):
    print('Reading {}'.format(directory.split("/")[-1]))

    # Read the entire csv file of the user:
    with gzip.open(directory,'rb') as fid:
        csv_str = fid.read();
        csv_str = csv_str.decode("utf-8")
        pass;

    (feature_names,label_names) = parse_header_of_csv(csv_str);
    n_features = len(feature_names);
    (X,Y,M,timestamps) = parse_body_of_csv(csv_str,n_features);

    return (X,Y,M,timestamps,feature_names,label_names);

def count_unique_consecutive(arr): # Count unique sequential labels
    indices=[0]
    for i in range(1,len(arr)):
        if not np.array_equal(arr[i],arr[i-1]):
            indices.extend([i])
    return indices,len(indices)


k_imputed_data='dataset/Rimpute/ximpute_kalman/'
ak_imputed_data='dataset/Rimpute/ximpute_arima_kalman/'

temporal_fit='dataset/temporal_fitting/'
uniqueind_temporalfitting=temporal_fit+'y_unique/'

k_globs=glob.glob(k_imputed_data)
ak_globs=glob.glob(ak_imputed_data)

# 0-127 are continuous features, 128- are discrete
feat=128


# In[ ]:


# Anomaly detection pipeline
def anomaly_detection(raw_data,fname,n_feat=feat):
    vanilla_peaks={} # Using SST getting just index
    for ind in range(n_feat):
        temp_data=raw_data[:,ind] # Going per feature/column

        model=banpei.SST(w=50)
        anomaly=model.detect(temp_data,is_lanczos=True) # Changepoint scores using SST
        peak=sig.find_peaks(anomaly) # Data peaks- naive

        vanilla_peaks[ind]=[peak[0],anomaly] # As dict --> Index values, actual scores
        print("\t\t\t Anomaly detection Index {} done.".format(ind))

    vanilla_fname=fname+'_vanillasst.csv'

    with open(vanilla_fname,'w') as f:
        writer=csv.writer(f)
        for k,v in vanilla_peaks.items():
            writer.writerow([k,v])
    f.close()
    print("\t\t Saved anomaly detection")


# In[ ]:


# Changepoint detection pipeline
def breakpoint_detection(raw_data,fname,estimated_breaks,n_feat=feat):
    ruptures_cpts={} # PELT,BinSeg,Dynp
    bocpd_l1={} # BOCPD l=200
    bocpd_l2={} # BOCPD l=400
    exo_cpd_offline={} #EXO CPD offline
    exo_cpd_online={} #EXO CPD online
    for ind in range(n_feat):
        temp_data=raw_data[:,ind] # Going per feature/column

        start=time.time()
        # Ruptures
        rpt_pelt=rpt.Pelt(model='rbf').fit(temp_data)
        pelt_result=rpt_pelt.predict(pen=5)
        print("Pelt: ",time.time()-start)

        start=time.time()
        rpt_binseg=rpt.Binseg(model='rbf').fit(temp_data)
        bin_result=rpt_binseg.predict(n_bkps=estimated_breaks)
        print("Binseg: ",time.time()-start)

#         start=time.time()
#         rpt_dynp=rpt.Dynp(model='normal',min_size=2,jump=5).fit(temp_data)
#         dynp_result=rpt_dynp.predict(n_bkps=estimated_breaks)
#         print("Dynp: ",time.time()-start)


#         ruptures_cpts[ind]=list(set().union(pelt_result,bin_result,dynp_result))
        ruptures_cpts[ind]=list(set().union(pelt_result,bin_result))

        #BOCPD
        start=time.time()
        hazard_func_l1=lambda r: bcp.constant_hazard(r, _lambda=200)
        beliefs_l1,maxes_l1=bcp.inference(temp_data, hazard_func_l1)
        log_bel_l1=-np.log(beliefs_l1)
        index_changes_l1=np.where(np.diff(maxes_l1.T[0])<0)[0]
        print("BOCPD_l1: ",time.time()-start)


        bocpd_l1[ind]=[index_changes_l1,log_bel_l1]

        start=time.time()
        hazard_func_l2=lambda r: bcp.constant_hazard(r, _lambda=400)
        beliefs_l2,maxes_l2=bcp.inference(temp_data, hazard_func_l2)
        log_bel_l2=-np.log(beliefs_l2)
        index_changes_l2=np.where(np.diff(maxes_l2.T[0])<0)[0]
        print("BOCPD_l2: ",time.time()-start)

        bocpd_l2[ind]=[index_changes_l2,log_bel_l2]

        #Offline/Online Exact and Efficient Bayesian Inference
        #Offline
#         start=time.time()
#         Q,P,Pcp = offcd.offline_changepoint_detection(temp_data,partial(offcd.const_prior, l=(len(temp_data)+1)), offcd.gaussian_obs_log_likelihood, truncate=-40)
#         offline_cpts=data=np.exp(Pcp).sum(0)
#         offline_peaks=find_peaks(offline_cpts)
#         print("Offline EXO: ",time.time()-start)


#         exo_cpd_offline[ind]=[offline_peaks,offline_cpts]

        #Online
        start=time.time()
        Nw=10
        R,maxes=oncd.online_changepoint_detection(temp_data, partial(oncd.constant_hazard, 250), oncd.StudentT(0.1, .01, 1, 0))
        online_cpts=R[Nw,Nw:-1]
        online_peaks=sig.find_peaks(online_cpts)
        print("Online EXO: ",time.time()-start)

        exo_cpd_online[ind]=[online_peaks,online_cpts]

        print("\t\t\t Breakpoint detection Index {} done.".format(ind))

    ruptures_fname=fname+'_ruptures.csv'
    bocpdl1_fname=fname+'_bocpdl1.csv'
    bocpdl2_fname=fname+'_bocpdl2.csv'
    ofexo_fname=fname+'_ofexo.csv'
    onexo_fname=fname+'_onexo.csv'


    with open(ruptures_fname,'w') as f:
        writer=csv.writer(f)
        for k,v in ruptures_cpts.items():
            writer.writerow([k,v])
    f.close()

    with open(bocpdl1_fname,'w') as f:
        writer=csv.writer(f)
        for k,v in bocpd_l1.items():
            writer.writerow([k,v])
    f.close()

    with open(bocpdl2_fname,'w') as f:
        writer=csv.writer(f)
        for k,v in bocpd_l2.items():
            writer.writerow([k,v])
    f.close()

#     with open(ofexo_fname,'w') as f:
#         writer=csv.writer(f)
#         for k,v in exo_cpd_offline.items():
#             writer.writerow([k,v])
#     f.close()

    with open(onexo_fname,'w') as f:
        writer=csv.writer(f)
        for k,v in exo_cpd_online.items():
            writer.writerow([k,v])
    f.close()

    print("\t\t Saved changepoint detection")


# In[ ]:
print("Files in current directory are: ",os.listdir())

for g in glob.glob("*.csv"):
    fname=g.split('/')[-1].split('_')[0]
    tar_fname=fname+'.features_labels.csv.gz'

    x_user,y_user,missedlabel_user,tstamp_user,featurename_user,labelname_user=read_user_data(tar_fname)

    sample_indices,sample_unique_len=count_unique_consecutive(y_user)
    print("\t\t Fname: {} >>> Original Length: {} --> Sequential unique: {}".format(fname,len(y_user),sample_unique_len))
    uniqueind_fname=fname+'_unique_yind.csv'
    np.savetxt(uniqueind_fname,sample_indices,fmt="%d",delimiter=",")
    print("\t\t Saving unique y-label indices/breakpoints.")

    # Reading the csvs from the actual imputed data, zero-impute any still remaining missing data
    raw_data=pd.read_csv(g,header=0,index_col=0).fillna(0).values

    anomaly_detection(raw_data=raw_data,fname=fname,n_feat=feat)
    print("\t\tAnomaly Detection done.")

    breakpoint_detection(raw_data=raw_data,fname=fname,estimated_breaks=sample_unique_len,n_feat=feat)
    print("\t\tBreakpoint Detection done.")
