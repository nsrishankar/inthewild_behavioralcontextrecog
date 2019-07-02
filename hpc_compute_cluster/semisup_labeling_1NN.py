#!/usr/bin/env python
# coding: utf-8

# Required imports
import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import gzip
import glob
import pickle
import copy
import math
from io import StringIO
import importlib.machinery
import time
from sklearn.neighbors import KNeighborsClassifier as KNN

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

def summarize_features(feature_list):
    summary_feature_list=np.empty_like(feature_list)
    for (ind,feature) in enumerate(feature_list):
        if feature.startswith('raw_acc'):
            summary_feature_list[ind]='phone_acc'
        if feature.startswith('proc_gyro'):
            summary_feature_list[ind]='phone_gyro'
        if feature.startswith('raw_magnet'):
            summary_feature_list[ind]='phone_mag'
        if feature.startswith('watch_acc'):
            summary_feature_list[ind]='watch_acc'
        if feature.startswith('watch_heading'):
            summary_feature_list[ind]='watch_dir'
        if feature.startswith('location'):
            summary_feature_list[ind]='phone_loc'
        if feature.startswith('audio'):
            summary_feature_list[ind]='phone_audio'
        if feature.startswith('discrete:app_state'):
            summary_feature_list[ind]='phone_app'
        if feature.startswith('discrete:battery'):
            summary_feature_list[ind]='phone_battery'
        if feature.startswith('discrete:on'):
            summary_feature_list[ind]='phone_use'
        if feature.startswith('discrete:ringer'):
            summary_feature_list[ind]='phone_callstat'
        if feature.startswith('discrete:wifi'):
            summary_feature_list[ind]='phone_wifi'
        if feature.startswith('lf'):
            summary_feature_list[ind]='phone_lf'
        if feature.startswith('discrete:time'):
            summary_feature_list[ind]='phone_time'

    return summary_feature_list

# Get a summary of the sensor feature along with the original label that was used
def summarize_features_worig(feature_list):
    summary_feature_list=np.empty((len(feature_list),2),dtype=object)

    for (ind,feature) in enumerate(feature_list):
        if feature.startswith('raw_acc'):
            summary_feature_list[ind,0]='phone_acc'
            summary_feature_list[ind,1]=feature

        if feature.startswith('proc_gyro'):
            summary_feature_list[ind,0]='phone_gyro'
            summary_feature_list[ind,1]=feature

        if feature.startswith('raw_magnet'):
            summary_feature_list[ind,0]='phone_mag'
            summary_feature_list[ind,1]=feature

        if feature.startswith('watch_acc'):
            summary_feature_list[ind,0]='watch_acc'
            summary_feature_list[ind,1]=feature

        if feature.startswith('watch_heading'):
            summary_feature_list[ind,0]='watch_dir'
            summary_feature_list[ind,1]=feature

        if feature.startswith('location'):
            summary_feature_list[ind,0]='phone_loc'
            summary_feature_list[ind,1]=feature

        if feature.startswith('audio'):
            summary_feature_list[ind,0]='phone_audio'
            summary_feature_list[ind,1]=feature

        if feature.startswith('discrete:app_state'):
            summary_feature_list[ind,0]='phone_app'
            summary_feature_list[ind,1]=feature

        if feature.startswith('discrete:battery'):
            summary_feature_list[ind,0]='phone_battery'
            summary_feature_list[ind,1]=feature

        if feature.startswith('discrete:on'):
            summary_feature_list[ind,0]='phone_use'
            summary_feature_list[ind,1]=feature

        if feature.startswith('discrete:ringer'):
            summary_feature_list[ind,0]='phone_callstat'
            summary_feature_list[ind,1]=feature

        if feature.startswith('discrete:wifi'):
            summary_feature_list[ind,0]='phone_wifi'
            summary_feature_list[ind,1]=feature

        if feature.startswith('lf'):
            summary_feature_list[ind,0]='phone_lf'
            summary_feature_list[ind,1]=feature

        if feature.startswith('discrete:time'):
            summary_feature_list[ind,0]='phone_time'
            summary_feature_list[ind,1]=feature

    return summary_feature_list

def choose_sensors(X_train,used_sensors,summarized_feature_names):
    used_sensor_feature_names=np.zeros(len(summarized_feature_names),dtype=bool)
    # Creates a zero boolean vector of all possible feature names
    for s in used_sensors:
        used_sensor_feature_names=np.logical_or(used_sensor_feature_names,(s==summarized_feature_names))
    X_train=X_train[:,used_sensor_feature_names]
    return X_train

def choose_sensors_dropout(X_train,used_sensors,summarized_feature_names):
    used_sensor_feature_names=np.zeros(len(summarized_feature_names),dtype=bool)
    data_length=len(X_train)

    # Creates a zero boolean vector of all possible feature names
    for s in used_sensors:
        used_sensor_feature_names=np.logical_or(used_sensor_feature_names,(s==summarized_feature_names))
    mask=np.tile(used_sensor_feature_names,(data_length,1))

    X_train=np.multiply(X_train,mask) # Element-wise matrix multiply
    return X_train

def choose_sensors_longnames(X_train,used_sensors,long_featurenames):

    used_sensor_feature_names=np.zeros(len(long_featurenames),dtype=bool)
    used_feature_actualnames=np.zeros(len(long_featurenames),dtype=bool)
    # Creates a zero boolean vector of all possible feature names
    summary_features=long_featurenames[:,0]
    all_complete_features=long_featurenames[:,-1]

    for s in used_sensors:
        similar=(s==summary_features)

        #used_complete_features=(all_complete_features[similar.astype(int)])

        used_sensor_feature_names=np.logical_or(used_sensor_feature_names,similar)
        used_feature_actualnames=np.logical_or(used_feature_actualnames,similar)

    X_train=X_train[:,used_sensor_feature_names]
    long_names=all_complete_features[used_feature_actualnames]
    return X_train,long_names

# Sensor Types, Label Possibilities variables
sensor_types=['phone_acc','phone_gyro','phone_mag','phone_loc','phone_audio',
'phone_app','phone_battery','phone_use','phone_callstat','phone_wifi','phone_lf',
'phone_time']
label_possibilities=['LOC_home','OR_indoors','PHONE_ON_TABLE','SITTING','WITH_FRIENDS',
 'LYING_DOWN','SLEEPING','WATCHING_TV','EATING','PHONE_IN_POCKET',
 'TALKING','DRIVE_-_I_M_A_PASSENGER','OR_standing','IN_A_CAR',
 'OR_exercise','AT_THE_GYM','SINGING','FIX_walking','OR_outside',
 'SHOPPING','AT_SCHOOL','BATHING_-_SHOWER','DRESSING','DRINKING__ALCOHOL_',
 'PHONE_IN_HAND','FIX_restaurant','IN_CLASS','PHONE_IN_BAG','IN_A_MEETING',
 'TOILET','COOKING','ELEVATOR','FIX_running','BICYCLING','LAB_WORK',
 'LOC_main_workplace','ON_A_BUS','DRIVE_-_I_M_THE_DRIVER','STROLLING',
 'CLEANING','DOING_LAUNDRY','WASHING_DISHES','SURFING_THE_INTERNET',
 'AT_A_PARTY','AT_A_BAR','LOC_beach','COMPUTER_WORK','GROOMING','STAIRS_-_GOING_UP',
 'STAIRS_-_GOING_DOWN','WITH_CO-WORKERS']

# Returns a standardized (0 mean, 1 variance) dataset
def standardize(X_train):
    mean=np.nanmean(X_train,axis=0).reshape((1,-1))# Ignores NaNs while finding the mean across rows
    standard_dev=np.nanstd(X_train,axis=0) # Ignores NaNs while finding the standard deviation across rows
    standard_dev_nonzero=np.where(standard_dev>0,standard_dev,1.).reshape((1,-1)) # Div zero

    X=(X_train-mean)/standard_dev_nonzero
    return X,mean,standard_dev_nonzero

def nearestneighbor(x,y):
    imputed_y=np.empty_like(y)
    for ind in range(y.shape[-1]):
        print("Current column: ",ind)
        y_col=y[:,ind]

        unique_col,counts_col=np.unique(y_col,return_counts=True)
        print(*zip(unique_col,counts_col))

        skip_col=0
        for i in range(len(unique_col)):
            if(unique_col[i]==-1):
                if counts_col[i]==len(y[:,ind]):
                    skip_col=1
                    print("Skipping column {}".format(ind))

        if (skip_col!=1):
            y_train_masked=ma.masked_where(y_col!=-1,y_col).mask
            missing_indices=np.where(y_train_masked==False)[0]


            x_train=x[y_train_masked]
            y_col_train=y_col[y_train_masked]

            knn_classifier=KNN(n_neighbors=1,weights='distance',p=2,metric='minkowski',n_jobs=-1)
            knn_classifier.fit(x_train,y_col_train)

            for i,index in enumerate(missing_indices):
                print("\t\tWorking on index {} of {}".format(i,len(missing_indices)))
                predict=knn_classifier.predict(x[index,:].reshape(1, -1))
                print(predict)
                #print("\t\t",knn_classifier.predict_proba(x_user[index,:].reshape(1, -1)))
                y_col[index]=predict
                y_train_masked=ma.masked_where(y_col!=-1,y_col).mask

                knn_classifier.fit(x[y_train_masked],y_col[y_train_masked])
                print("\t\tMissing length is now :",len(np.where(y_train_masked==False)[0]))
            imputed_y[:,ind]=y_col
    #unique,counts=np.unique(imputed_y,return_counts=True)
    #print(*zip(unique,counts))

    return imputed_y

for g in glob.glob("*.csv.gz"):
    fname=g.split('/')[-1].split('.')[0]

    x_user,y_user,missedlabel_user,tstamp_user,featurename_user,labelname_user=read_user_data(g)
    x_user,_,_=standardize(x_user)
    x_user=np.nan_to_num(x_user)
    feature_names=summarize_features_worig(featurename_user)
    x_user,feature_long_names=choose_sensors_longnames(x_user,sensor_types,feature_names)

    y_out=nearestneighbor(x_user,y_user)
    print("\t\tNearest neighbor done for user {}.".format(fname))

    print("Orig shape: {}--> New shape: {}".format(y_user.shape,y_out.shape))

    orig_unique,orig_counts=np.unique(y_user,return_counts=True)
    ss_unique,ss_counts=np.unique(y_out,return_counts=True)
    print("Orig:{} --> Impute:{}".format(*zip(orig_unique,orig_counts),*zip(ss_unique,ss_counts)))

    out_fname=fname+'_1NN.pkl'
    f=open(out_fname,'wb')
    pickle.dump(y_out,f)
    f.close()
