# Import to access files
import os
import glob

# to structure and transform the data
import pandas as pd
import numpy as np

# to plot / vizualize the data
import matplotlib.pyplot as plt
import seaborn as sns

# to perform computation on data (data feature extraction, statistical analysis...)
import scipy
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.integrate import simpson as simps

# to perform PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# to perform random forest machine learning modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# to visualize the results / output of the machine learning analysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

# for subject-wise treatment
from itertools import combinations

# for convenience
from functools import partial

import datetime

all_sensors = ["sensor_hip_CUT_data", "sensor_wrist_CUT_data", "sensor_pocket_CUT_data"]

def get_data_files(parent_dir,sensor,subjects):
    files = []
    basepath = os.path.join(parent_dir, "DataSample", "sensor_CUT_data",sensor)
    for subject in subjects:
        filelist = sorted(glob.glob(os.path.join(basepath,subject,"*.csv")))
        if len(filelist) < 1:
            print("WARNING:",sensor)
        files.append(filelist) #alphabetical order 
    return [item for sublist in files for item in sublist]  

_activities = ['run', 'walk', 'kick', 'bicycle', 'sit', 'stand','skate', 'upstairs', 'downstairs']
_activities_extra = ['slow_walk', 'fast_walk', 'slow_skate', 'fast_skate', 'slow_kick', 'fast_kick', 'jog']
_higher_level_def = {'bicycle': ['bicycle'], 'locomotive': ['jog', 'walk', 'upstairs', 'downstairs'], 
                 'push-push-glide': ['kick', 'skate'], 'sedentary': ['sit', 'stand']}
_higher_level = {a:k for k,v in _higher_level_def.items() for a in v}

def get_activity(file, activities):
    def helper_label(labels, f):
        label='Not found'
        for a in labels:
            if a in f:
                label=a
                break
        return label
    activity= helper_label(activities,os.path.basename(file)) 
    return activity

get_label = partial(get_activity, activities = _activities)

def get_higher_level_label(file, high_level=_higher_level):
    label = get_label(file)
    if label in high_level:
        return high_level[label]
    return label # was not found

def get_subject_label(file, data, **to_use):
    subject_char=os.path.split(os.path.dirname(file))[1]
    data_vals = data[data["subject_ID"]==subject_char]
    to_return = [subject_char]+[data_vals[k].iloc[0] for k in to_use if to_use[k]]
    return to_return

def nearest_value(input_list, input_value):
    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return arr[i]

    #PCAを使うための準備ここから〜
def get_feature(data, N, ExpWinLen):
    # 1. 必要な6軸の波形データを抽出
    raw_values = data[['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
    
    # 2. データを平坦化（2次元の波形を、1行の長いリストにする）
    feature = raw_values.flatten().tolist()
    
    # 3. 列名（カラム名）を自動生成（raw_0, raw_1, ...）
    name = [f"raw_{i}" for i in range(len(feature))]
    
    return feature, name
    #↑ここまで

# Parametres
# start = 500 # portion of data cut at the beggining of the file
# end = 500 # portion of data cut at the end of the file
# W_size = 10 # 10 sec
# timescale = 100 #working with 10 ms per step

def data_prep2(data, cut_start=500, cut_end=500):
    # Compute all over the original file (not within each window), Some parameters needed for further calculation of the features.
    ### calculate the Euclindienne norm -1 (needed for feature from Willetts)
    VECT = np.sqrt(pow(data['x-axis (g)'],2)+pow(data['y-axis (g)'],2)+pow(data['z-axis (g)'],2))

    ENMO = []
    for i in VECT:
        ENMO.append(i-1)

    ENMOTRUNC = []
    for en in ENMO:
        ENMOTRUNC.append(max(en,0))

    data["Vector"] = VECT
    data["Enmo"]= ENMO
    data["EnmoTrunc"] = ENMOTRUNC

    ### Needed for Feature taken from Ellis.
    ROLL=np.arctan2(data['y-axis (g)'],data['z-axis (g)']) # make sure I should not use arctan instead (see Skateboarding_Data).
    PITCH = np.arctan2(data['x-axis (g)'],data['z-axis (g)']) # same comment as above.
    YAW = np.arctan2(data['y-axis (g)'],data['x-axis (g)']) # same comment as above.

    data['Roll'] = ROLL # see comment about arctan vs. arctan2
    data['Pitch'] = PITCH #idem
    data['Yaw'] = YAW #idem
    
    ##
    ##
    
    ### Needed for some parameters suggested by Ellis et al. MSSE 2016.
    # separate the acceleration components due to gravity (GA) from the total  acceleration.
    # See Karantonis and al.  IEEE TRANSACTIONS ON INFORMATION TECHNOLOGY IN BIOMEDICINE 2006

    # Build a low pass filter
    # order= 3
    # cut-off frequency at 0.5 Hz (Ellis, Karantonis originally suggested 0.25)
    # passband ripple: 0.01 dB
    # stopband: −100 dB 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellip.html

    # Design filter 
    LPF = signal.ellip(3, 0.01, 100, 0.25, 'low', fs=100, output='sos')
    # 0.25Hz (Karantonis) vs 0.5 (Ellis) does not changed much.
    # Keep the original one from Karantonis, which is better described with order, rp and rs in the article.
    # according to the exemple in:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellip.html
    # the cut-off frequency is specified in Hz for digital filters.

    # Apply filter 
    FILT_x = signal.sosfilt(LPF, data['x-axis (g)'])
    FILT_y = signal.sosfilt(LPF, data['y-axis (g)'])
    FILT_z = signal.sosfilt(LPF, data['z-axis (g)'])

    data['Filt_x'] = FILT_x
    data['Filt_y'] = FILT_y
    data['Filt_z'] = FILT_z
    
    ##
    ##
    
    # remove the first and last 5 seconds (start and stop of each exp = not clean)
    data_cut = data[["epoch(s)","timestamp","elapsed (s)","merger",
               'x-axis (g)','y-axis (g)','z-axis (g)',
               "x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)",
               "Vector", "Enmo", "EnmoTrunc",
               "Roll","Pitch","Yaw",
               "Filt_x", "Filt_y", "Filt_z"]].copy()
    
    data_cut = data_cut.iloc[cut_start:-cut_end]
    
    # renaming columns according to my research notebook. So that I can use the function

    
    
    data_cut.rename(columns = {"epoch(s)": "epoch","elapsed (s)":"ElapsedTime",
                               'x-axis (g)':"acceleration_x",'y-axis (g)':"acceleration_y",'z-axis (g)':"acceleration_z",
                               "x-axis (deg/s)":"gyro_x", "y-axis (deg/s)":"gyro_y", "z-axis (deg/s)":"gyro_z"}, inplace = True)
    
    timeS=data_cut
    return timeS

def compute_features_for_all_windows(files, subject_data, W_size=10,timescale=100):
    # First I look at the maximal and minimal size window
    MaxSizeWin = 0
    MinSizeWin = 2*W_size*timescale

    count=0
    tot_windows = 0
    warnings = 0
    windows_per_activity = []
    FEAT=[]
    ACTIVITY = []
    TRUE_LABEL=[]
    SUBJECT=[]
    HIGHER_LEVEL_LABEL=[]
    for f in files:
        data = pd.read_csv(f)
        timeS = data_prep2(data) #second version of data prep
        count+=1
        countWin=0
        start_val=[round(timeS.iloc[0]["ElapsedTime"],3)]
        end_val=[]
        start_row=[0]
        end_row=[]
        row=0
        wincount=0
        # Not very useful with clean data, but checks data integrity as well
        j=timeS.iloc[0]["ElapsedTime"]-1
        
        for i in timeS["ElapsedTime"]:
            i = round(i,3)
            if i == round(start_val[0]+W_size*(wincount+1),3) and row < len(timeS["ElapsedTime"]): # Error raised if second condition is removed        
                end_val.append(j) # because j will have increamented before the criteria is meet.
                start_val.append(round(timeS.iloc[row]["ElapsedTime"],3))
                wincount+=1
                end_row.append(row)
                start_row.append(row)
            
            if i > start_val[0]+W_size*(wincount+1):
                #debug
                print(i,j,start_val[0]+W_size*(wincount+1),start_val[-1])
            row+=1 #faster version: W_size*timescale
            j=i
        start_val.pop() # remove the last orphaned value
        start_row.pop()
        Starter = start_row#start_val #[0:min(len(start_val),len(end_val))]
        Finisher = end_row #end_val #[0:min(len(start_val),len(end_val))]
    
        for a,b in zip(start_row,end_row):
            if b-a < W_size*timescale: #less than W_size
                print(f"Warning: window was too short ({b-a}), skipped")
                warnings += 1
                continue
            elif b-a >W_size*timescale: #too long
                print(f"Warning: window was too long ({b-a}), skipped")
                warnings +=1
                continue
            tot_windows+=1
            countWin+=1
        
            da = pd.DataFrame(timeS[a:b]) # I suggest to use .iloc to keep the terminology consistent with the one use elsewhere in the cell.
            # Add difference between elepsed and previous ellapsed.
        
            if da.shape[0] < MinSizeWin:
                MinSizeWin = da.shape[0]
            if da.shape[0] > MaxSizeWin:
                MaxSizeWin = da.shape[0]
        
            feature, name = get_feature(data=da, N=b-a, ExpWinLen=W_size)
            FEAT.append(feature)
            true_label =get_label(f)
            
            TRUE_LABEL.append(true_label)
        
            SUBJECT.append(get_subject_label(f,subject_data,sex=True,age_group=True,level=True,stance=True))
            HIGHER_LEVEL_LABEL.append(get_higher_level_label(f))
        ###
        windows_per_activity.append(countWin)

    TableFeature = pd.DataFrame(FEAT)
    TableFeature.columns = name
    TableFeature["true_label"] = TRUE_LABEL
    TableFeature["category"] = HIGHER_LEVEL_LABEL
    sbjts = np.array(SUBJECT)
    TableFeature["subject"] = sbjts[:,0]
    TableFeature["sex"] = sbjts[:,1]
    TableFeature["age_group"] = sbjts[:,2]
    TableFeature["level"] = sbjts[:,3]
    TableFeature["stance"] = sbjts[:,4]
    return TableFeature

def save_data(TableFeature, sensor, directory, acceleration_only=False, last_acc="VectRect_mean", num_info=7):
    # Metadata
    ActName=["nb of windows of Kickboard","nb of windows of Skateboard","nb of windows of Run","nb of windows of Walk", "Total number of windows"]
    Windows=[
         TableFeature.loc[TableFeature["true_label"] == "kick"].shape[0],
         TableFeature.loc[TableFeature["true_label"] == "skate"].shape[0],
         TableFeature.loc[TableFeature["true_label"] == "run"].shape[0],
         TableFeature.loc[TableFeature["true_label"] == "walk"].shape[0]]
    Windows.append(sum(Windows))
    winwin=pd.DataFrame(Windows,ActName)

    if acceleration_only:
        pos = TableFeature.columns.get_loc(last_acc)
        TableFeature.drop(columns=TableFeature.columns[pos+1:- num_info], inplace=True)
        sensor = sensor+"_Acc"
    
    if not os.path.exists(directory):
        os.mkdir(directory)
    if not os.path.isdir(directory):
        print(f"Warning, {directory} already exists")
    winwin.to_csv(os.path.join(directory,str(datetime.datetime.now().isoformat(timespec='seconds')).replace(":","_")+"_Nb_of_windows_"+sensor+".csv"))
    TableFeature.to_csv(os.path.join(directory,str(datetime.datetime.now().isoformat(timespec='seconds')).replace(":","_")+"_DataFeatures_"+sensor+".csv"))
    
if __name__ == "__main__":
    # Subject data; should be implemented as external data, e.g., csv
    InfoSubIndex =['subject_ID', 'sex', 'age', 'age_group', 'level', 'stance', 'weight'] 
    InfoSub1 = ["001", 'woman', 12, "kid", "expert", "regular",38] # Pivot OK
    InfoSub2 = ["002",'woman', 26, "young adult", "beginner", "goofy",42] # Pivot OK
    InfoSub3 = ["003",'man', 40, "young adult", "expert", "regular",83] # Pivot OK
    InfoSub4 = ["004",'woman', 27, "young adult", "expert", "regular",45] # No pivot when cruising (video)
    InfoSub5 = ["005",'man', 22, "young adult", "beginner", "regular", 54] # Pivot OK
    InfoSub6 = ["006",'man', 22, "young adult", "expert", "goofy",65] # weight: rough estimation
    InfoSub7 = ["007",'man', 13, "kid", "expert", "regular",50] # weight: rough estimation
    InfoSub8 = ["008",'man', 25, "young adult", "expert","goofy",80] # weight: rough estimation
    InfoSub9 = ["009",'man', 22, "young adult", "expert", "regular",65] # weight: rough estimation
    InfoSub10 = ["010",'man', 55, "older adult", "beginner", "regular",65] # weight: rough estimation

    SubjectCharacteristics = pd.DataFrame([InfoSub1,InfoSub2,InfoSub3,InfoSub4,InfoSub5,InfoSub6,InfoSub7,InfoSub8,InfoSub9,InfoSub10],columns=InfoSubIndex)
    subjects = list(SubjectCharacteristics["subject_ID"])
    current = os.getcwd()
    parent = os.path.abspath(os.path.join(current, os.pardir))
    prepare_acc_only = ["sensor_hip_CUT_data", "sensor_wrist_CUT_data"]

    for sensor in all_sensors:
        files = get_data_files(parent,sensor,subjects)
        win_size=20
        TF = compute_features_for_all_windows(files, SubjectCharacteristics, W_size=win_size)
        save_data(TF,sensor,os.path.join(parent,f"FeatureDataFrames_{win_size}"))
        if sensor in prepare_acc_only:
            save_data(TF,sensor,os.path.join(parent,f"FeatureDataFrames_{win_size}"),acceleration_only=True)
