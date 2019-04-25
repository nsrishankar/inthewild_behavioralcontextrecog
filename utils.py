import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from numpy.lib.stride_tricks import as_strided
from scipy.interpolate import UnivariateSpline

# Confusion Matrix for multi-label datasets
def cm(yt, yp, classes,fname):
    instcount = yt.shape[0]
    n_classes = classes.shape[0]
    mtx = np.zeros((n_classes, 4))
    for i in range(instcount):
        for c in range(n_classes):
            mtx[c,0] += 1 if yt[i,c]==1 and yp[i,c]==1 else 0
            mtx[c,1] += 1 if yt[i,c]==1 and yp[i,c]==0 else 0
            mtx[c,2] += 1 if yt[i,c]==0 and yp[i,c]==0 else 0
            mtx[c,3] += 1 if yt[i,c]==0 and yp[i,c]==1 else 0
    mtx = [[m0/(m0+m1), m1/(m0+m1), m2/(m2+m3), m3/(m2+m3)] for m0,m1,m2,m3 in mtx]
    plt.figure(num=None, figsize=(30, 15), dpi=100, facecolor='w', edgecolor='k')
    plt.imshow(mtx, interpolation='nearest',cmap='Blues')
    plt.title("title")
    tick_marks = np.arange(n_classes)
    plt.xticks(np.arange(4), ['1 - 1','1 - 0','0 - 0','0 - 1']) # Real-predicted
    plt.yticks(tick_marks, classes)
    for i, j in itertools.product(range(n_classes), range(4)):
        plt.text(j, i, round(mtx[i][j],2), horizontalalignment="center")

    plt.tight_layout()
    plt.ylabel('Labels')
    plt.xlabel('Predicted')
    plt.savefig(fname)
    plt.show()

    
# Remove outliers in a dataset: 25-75 percentile, Q1/Q3-+1.5*IQR, percentage, zscore/n- standard deviations away 
def remove_outliers(df_x,df_y,method):
    method=method.split('_')
    outlier_indices_all=[]
    
    if method[0]=='iqr':
        for i in range(df_y.shape[-1]): # Loop across all labels
            temp=df_x[df_y.iloc[:,i]==1] # Get datapoints in X-dataset where the label is a 1
            initial_shape=temp.shape

            per_label_indices=[]
            if initial_shape[0]!=0: # Have examples for that label
                for j in range(initial_shape[-1]): # Loop through columns in the X-dataset
                    temp_col=temp.iloc[:,j]
                    quantile1=temp_col.quantile(0.25)
                    quantile3=temp_col.quantile(0.75)
                    
                    quantile1_df=temp_col.index[temp_col<quantile1]
                    quantile3_df=temp_col.index[temp_col>quantile3]
                    indices=quantile1_df.union(quantile3_df)
                    per_label_indices.extend(indices.values.tolist())
                    
            per_label_indices=list(set(per_label_indices))
            outlier_indices_all.extend(per_label_indices)
            
    elif method[0]=='iqrwhiskers':
        for i in range(df_y.shape[-1]): # Loop across all labels
            temp=df_x[df_y.iloc[:,i]==1] # Get datapoints in X-dataset where the label is a 1
            initial_shape=temp.shape

            per_label_indices=[]
            if initial_shape[0]!=0: # Have examples for that label
                for j in range(initial_shape[-1]): # Loop through columns in the X-dataset
                    temp_col=temp.iloc[:,j]
                    quantile1=temp_col.quantile(0.25)
                    quantile3=temp_col.quantile(0.75)
                    iqr=quantile3-quantile3-quantile1
                    
                    
                    quantile1_df=temp_col.index[temp_col<quantile1-1.5*iqr]
                    quantile3_df=temp_col.index[temp_col>quantile3+1.5*iqr]
                    indices=quantile1_df.union(quantile3_df)
                    per_label_indices.extend(indices.values.tolist())
                    
            per_label_indices=list(set(per_label_indices))
            outlier_indices_all.extend(per_label_indices)
    
    elif method[0]=='percent':
        percent_outlier=float(method[-1]) # Low outlier
       
        for i in range(df_y.shape[-1]): # Loop across all labels
            temp=df_x[df_y.iloc[:,i]==1] # Get datapoints in X-dataset where the label is a 1
            initial_shape=temp.shape

            per_label_indices=[]
            if initial_shape[0]!=0: # Have examples for that label
                for j in range(initial_shape[-1]): # Loop through columns in the X-dataset
                    temp_col=temp.iloc[:,j]
                    percent_low=temp_col.quantile(percent_outlier)
                    percent_high=temp_col.quantile(1.-percent_outlier)
    
                    percent_low_df=temp_col.index[temp_col<percent_low]
                    percent_high_df=temp_col.index[temp_col>percent_high]
                    indices=percent_low_df.union(percent_high_df)
                    per_label_indices.extend(indices.values.tolist())
                    
            per_label_indices=list(set(per_label_indices))
            outlier_indices_all.extend(per_label_indices)
        
    elif method[0]=='zscore':
        zscore_val=float(method[-1])
    
        for i in range(df_y.shape[-1]): # Loop across all labels
            temp=df_x[df_y.iloc[:,i]==1] # Get datapoints in X-dataset where the label is a 1
            initial_shape=temp.shape

            per_label_indices=[]
            if initial_shape[0]!=0: # Have examples for that label
                for j in range(initial_shape[-1]): # Loop through columns in the X-dataset
                    # Indices where values in column in temp is more than 3 standard deviations away
                    indices=np.where(np.absolute(zscore(temp.iloc[:,j])>zscore_val))[0] # Remove outliers
                    per_label_indices.extend(indices)
            per_label_indices=list(set(per_label_indices)) # Unique indices
            outlier_indices_all.extend(per_label_indices)

    outlier_indices_all=list(set(outlier_indices_all)) # All outliers across all features for all labels
    keep_indices=set(range(df_x.shape[0]))-set(outlier_indices_all)
    print("Originally {} datapoints.Removing {} datapoints".format(df_x.shape[0],len(outlier_indices_all)))

    return df_x.take(list(keep_indices)).reset_index(),df_y.take(list(keep_indices)).reset_index()

# N-Window moving average filters
def movingaverage(feature_array,feature_names,avg_diff,chosen_featurelist,isnan=0):
    df_features=pd.DataFrame(feature_array,columns=feature_names)
    
    # Window sizes
    minute_5=math.floor((60*5)/avg_diff)
    minute_30=math.floor((60*30)/avg_diff)
    hour_1=math.floor((60*60)/avg_diff)
    hour_3=math.floor((60*60*3)/avg_diff)
    hour_5=math.floor((60*60*5)/avg_diff)
    hour_10=math.floor((60*60*10)/avg_diff)
    day_1=math.floor((60*60*24)/avg_diff)
    day_2=math.floor((60*60*24*2)/avg_diff)
    
    windows=[minute_5,minute_30,hour_1,hour_3,hour_5,hour_10,day_1,day_2]
    i=0
    for w in windows:
        if (w<=1): # If the window size is calculated to be zero (no sliding) or one(sliding by every element)
            w=2 # Set it to every other element (window=2)
        for feat in chosen_featurelist:
#             print(w)
            temp_series=df_features.iloc[:,feat]
            if isnan==0:
                temp=temp_series.rolling(window=w,min_periods=1).mean() # To skip NaNs
            else:
                temp=temp_series.rolling(window=w).mean() # Will have w-1 nan values for each window
            name='MA_{}_{}'.format(feature_names[feat],str(i))
            df_features[name]=temp
#             print(df_features.shape)
            del temp
            del temp_series
            i+=1
    return df_features.values

# N-Window moving average filters that is exponentially weighted
def movingaverage_weighted(feature_array,feature_names,avg_diff,chosen_featurelist):
    df_features=pd.DataFrame(feature_array,columns=feature_names)
    
    # Window sizes
    minute_5=math.floor((60*5)/avg_diff)
    minute_30=math.floor((60*30)/avg_diff)
    hour_1=math.floor((60*60)/avg_diff)
    hour_3=math.floor((60*60*3)/avg_diff)
    hour_5=math.floor((60*60*5)/avg_diff)
    hour_10=math.floor((60*60*10)/avg_diff)
    day_1=math.floor((60*60*24)/avg_diff)
    day_2=math.floor((60*60*24*2)/avg_diff)
    
    windows=[minute_5,minute_30,hour_1,hour_3,hour_5,hour_10,day_1,day_2]
    i=0
    for w in windows:
        if (w<=1): # If the window size is calculated to be zero (no sliding) or one(sliding by every element)
            w=2 # Set it to every other element (window=2)
        for feat in chosen_featurelist:
#             print(w)
            temp_series=df_features.iloc[:,feat]
            
            alpha=0.85
            ewma=lambda x: pd.Series(x).ewm(alpha=alpha,adjust=False).mean().iloc[-1]
            temp=temp_series.rolling(window=w,min_periods=1).apply(ewma)
#             temp=temp_series.rolling(window=w,min_periods=1).mean() # To skip NaNs
        
            name='MA_{}_{}'.format(feature_names[feat],str(i))
            df_features[name]=temp
#             print(df_features.shape)
            del temp
            del temp_series
            i+=1
    return df_features.values       


def windowed_view(arr, window, overlap):
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                  window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                   arr.strides[-1:])
    return as_strided(arr, shape=new_shape, strides=new_strides)

def extrapolate(arr,desired_len):
    old_indices=np.arange(0,len(arr))
    new_indices=np.linspace(0,len(arr)-1,desired_len)
    spl=UnivariateSpline(old_indices,arr,k=3,s=0)
    return spl(new_indices)

def movingavg_overlaps(feature_array,feature_names,overlap_per,avg_diff,chosen_featurelist):
    df_features=pd.DataFrame(feature_array,columns=feature_names)
    
    # Window sizes
    minute_5=math.floor((60*5)/avg_diff)
    minute_30=math.floor((60*30)/avg_diff)
    hour_1=math.floor((60*60)/avg_diff)
    hour_3=math.floor((60*60*3)/avg_diff)
    hour_5=math.floor((60*60*5)/avg_diff)
    hour_10=math.floor((60*60*10)/avg_diff)
    day_half=math.floor((60*60*12)/avg_diff)
    day_1=math.floor((60*60*24)/avg_diff)
    day_2=math.floor((60*60*24*2)/avg_diff)
    
    windows=[minute_5,minute_30,hour_1,hour_3,hour_5,hour_10,day_half]
#     windows=[minute_5,minute_30,hour_1,hour_3,hour_5,hour_10,day_1,day_2]
    i=0
    for w in windows:
        if (w<=1): # If the window size is calculated to be zero (no sliding) or one(sliding by every element)
            w=2 # Set it to every other element (window=2)
        overlap_len=int(np.floor(w*overlap_per))
        for feat in chosen_featurelist:

            temp_series=df_features.iloc[:,feat].values
            window_avg=np.mean(windowed_view(temp_series,w,overlap_len),axis=-1)
            print("Extrapolating: ",len(window_avg)," to ",len(temp_series))
            extrapolated_window_avg=extrapolate(window_avg,len(temp_series)) # Crude extrapolation
            name='MA_{}_{}'.format(feature_names[feat],str(i))
            
            df_features[name]=extrapolated_window_avg
            del window_avg
            del temp_series
            i+=1
    return df_features.values