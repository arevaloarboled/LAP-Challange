# %% [markdown]
# # Libraries

# %%
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import dill

import glob, os
import gc
from inspect import isfunction

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, activations, backend
from sklearn.decomposition import PCA

import pandas as pd
import seaborn as sns

# !pip install spatialaudiometrics
from spatialaudiometrics import lap_challenge
from spatialaudiometrics import load_data
from spatialaudiometrics import hrtf_metrics
# from task2_create_sparse_hrtf import create_sparse_hrtf
# !pip install sofar
import sofar
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

# %% [markdown]
# # Functions
# 

# %%
# Design a low pass filter
def lowpass_filter(data, cutoff_freq=3000, fs=48000, order=10):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = scipy.signal.filtfilt(b, a, data)
    return filtered_data

# code from
def itd_estimator_maxiacce(hrir,fs):
    '''
    Calculates the ITD based on the MAXIACCe mode (transcribed from the itd_estimator in the AMTtoolbox 20/03/24, based on Andreopoulou et al. 2017)

    :param hrir: 3d array of the impulse response at each location x ear. Shape should be locations x ears x samples
    :return itd_s: ITD in seconds for each location
    :return itd_samps: ITD in samples for each location
    :return maxiacc: The max interaural cross correlation calculated
    '''
    itd_samps   = list()
    maxiacc     = list()
    for loc in hrir:
        # Take the maximum absolute value of the cross correlation between the two ears to get the maxiacc
        correlation     = scipy.signal.correlate(np.abs(scipy.signal.hilbert(loc[0,:])),np.abs(scipy.signal.hilbert(loc[1,:])))
        maxiacc.append(np.max(np.abs(correlation)))
        idx_lag         = np.argmax(np.abs(correlation))
        itd_samps.append(idx_lag - np.shape(hrir)[2])
    itd_samps=np.array(itd_samps)
    itd_s = itd_samps/fs

    return itd_s,itd_samps,maxiacc

def rotate_vect(x, nr):
    """
    Right circular rotation of x by nr places, thus y[nr] = x[0]
    and y[nr-1] = x[-1].
    
    :param x: numpy array, the vector to be rotated
    :param nr: int, number of places to rotate the vector to the right
    :return: numpy array, the rotated vector
    """
    n = len(x)
    if nr <= 0:
        nr = n + nr
    y = np.zeros_like(x)
    y[:nr] = x[-nr:]
    y[nr:] = x[:-nr]
    return y
def matlab_rceps(y,n):
    """
    Process a signal using a specific window function and inverse Fourier transform.
    
    :param y: numpy array, the input signal
    :param n: int, length of the signal (must be the same as len(y))
    :return: numpy array, processed signal
    """
    Y=np.real(np.fft.ifft(np.log(abs(np.fft.fft(y)))));
    # Define the window w
    w = np.concatenate([
        np.array([1]),                  # Starting element
        2 * np.ones(n // 2 - 1),        # Middle elements, twice the value of ones
        np.ones(1 - n % 2),             # Possibly one element, depending on n being odd or even
        np.zeros(n // 2 - 1)            # Ending elements
    ])
    # Apply the window to the signal and take the Fourier transform
    w_y = w * Y
    w_y_fft = np.fft.fft(w_y)
    # Exponential of the Fourier transformed windowed signal
    exp_w_y_fft = np.exp(w_y_fft)
    # Compute the real part of the inverse Fourier transform
    ym = np.real(np.fft.ifft(exp_w_y_fft))
    return ym
#convert a impulse response to the minimum phase filter
#this funciton can receive either the magnitud of an HRTF or an HRIR
def minPhaseHRIR(h,is_HRTF=True):
    H=[]
    if not is_HRTF:
        H = abs(np.fft.fft(h)) # FFT of the input signal
    else:
        H = np.abs(np.concatenate((h,h[-2:0:-1])))
    H=np.real(np.fft.ifft(H))
    H=rotate_vect(H,len(H)//2)
    return matlab_rceps(H,len(H))

#an array full of impulse responses are converted to minimum phase
def back_from_mag(h_i):
    return np.apply_along_axis(minPhaseHRIR,len(h_i.shape)-1,h_i)

#create layers as specified
def make_layer(li,name):
    ley=None
    if isinstance(li, list):
        if len(li)==2:
            if isinstance(li[1],tuple):
                li[1][1]["activation"]=li[1][0]; li[1][1]["name"]=name+"_dense"
                ley=layers.Dense(li[0],**li[1][1])
            else:
                ley=layers.Dense(li[0],activation=li[1],name=name+"_dense")
        else:
            #filters, kernel, strides, padding, activaiton, group or depth_multiplier
            defecto=[1,"valid",None,1]
            for j in range(len(li),7):
                li.append(defecto[j-3])
            if li[0]==3:
                ley=layers.Conv3D(li[1],li[2],li[3],li[4],activation=li[5],groups=li[6],use_bias=True,name=name+"_3d")
            if li[0]==2:
                ley=layers.Conv2D(li[1],li[2],li[3],li[4],activation=li[5],groups=li[6],use_bias=True,name=name+"_2d")
            if li[0]==1:
                ley=layers.Conv1D(li[1],li[2],li[3],li[4],activation=li[5],groups=li[6],use_bias=True,name=name+"_1d")
            if li[0]==-3:
                ley=layers.Conv3DTranspose(li[1],li[2],li[3],li[4],activation=li[5],use_bias=True,name=name+"_3dt")
            if li[0]==-2:
                ley=layers.Conv2DTranspose(li[1],li[2],li[3],li[4],activation=li[5],use_bias=True,name=name+"_2dt")
            if li[0]==-1:
                ley=layers.Conv1DTranspose(li[1],li[2],li[3],li[4],activation=li[5],use_bias=True,name=name+"_1dt")
            if li[0]==22:
                ley=layers.SeparableConv2D(li[1],li[2],li[3],li[4],activation=li[5],depth_multiplier=li[6],use_bias=True,name=name+"_2d")
    elif isinstance(li, float):
        if li<0:
            ley=layers.GaussianDropout(abs(li),name=name+"_drop")
        else:
            ley=layers.GaussianNoise(li,name=name+"_noise")
    elif isinstance(li, tuple):
        if li[0]>0:
            ley=layers.Reshape(li,name=name+"_reshape")
        else:
            if len(li[1])==3:
                ley=layers.Cropping3D(li[1],name=name+"_3dcrop")
            if len(li[1])==2:
                ley=layers.Cropping2D(li[1],name=name+"_2dcrop")
            if len(li[1])==1:
                ley=layers.Cropping1D(li[1],name=name+"_1dcrop")        
    elif isfunction(li):
        ley=layers.Lambda(li,name=name+"_func")
    else:
        if li=='f':
            ley=layers.Flatten(name=name+"_flat")
        else:
            ley=layers.BatchNormalization(name=name+"_Bnorm")
    return ley

#build model, this function is intended to create autoencoders
def build_model(shp,l,name="model"):
    autoco,enco,deco=None,None,None
    l_in=keras.Input(shape=shp,name=name+"_input"); lay=None; c=0
    for i in l[0]:
        lay=make_layer(i,name+"_enco_"+str(c))(lay if lay is not None else l_in); c+=1
    for i in l[1]:
        lay=make_layer(i,name+"_deco_"+str(c))(lay if lay is not None else l_in); c+=1
    autoco=keras.Model(l_in,lay)
    
    e_in=keras.Input(shp)
    e_l = None
    for i in range(0,len(autoco.layers)):
        if autoco.layers[i].name.split("_")[1]=="enco":
            e_l = autoco.layers[i](e_l if e_l is not None else e_in)
    if e_l is not None:
        enco=keras.Model(e_in,e_l)
    
    if enco!=None:
        d_in=keras.Input(e_l.shape[1:])
        d_l = None
        for i in range(0,len(autoco.layers)):
            if autoco.layers[i].name.split("_")[1]=="deco":
                d_l = autoco.layers[i](d_l if d_l is not None else d_in)
        if d_l is not None:
            deco=keras.Model(d_in,d_l)
    
    return (autoco,enco,deco)

def flip_azis(p):
    return np.lexsort((np.mod(360-p[:,0],360),p[:,1]))

# %% [markdown]
# # Load data
# This requires to have the `FreeFieldCompMinPhase_NoITD_48kHz` subjects in a folder called `SONICOM_only`

# %%
itds=[]; pos=[]; ir=[]; sr=0; sofas=[] #f_names=[]
for file_path in sorted(glob.glob("SONICOM_only/*.sofa")):
    sofa = sofar.read_sofa(file_path,verbose=0)
    sofas.append(file_path)
    ir.append(sofa.Data_IR); pos=sofa.SourcePosition; itds.append(sofa.Data_Delay); sr=sofa.Data_SamplingRate;
itds=np.array(itds); ir=np.array(ir) #convert to numpy arrays
itds=itds[:,:,1]-itds[:,:,0] #get itds
tmp=np.lexsort((pos[:,0],pos[:,1])); itds=itds[:,tmp]; ir=ir[:,tmp,:,:]; pos=pos[tmp,:] #sort by locations
tmp=np.array([True]*ir.shape[1]); tmp[np.where((pos[:,1]==90))[0][1:]]=False; ir=ir[:,tmp,:,:]; itds=itds[:,tmp]; pos=pos[tmp,:] # delete all the repeated zenith and only keep one
idx_all = np.random.permutation(len(ir)); idx=np.array([False]*len(ir)); idx[idx_all[0:round(len(ir)*0.8)]]=True #split locaitons 80 20 for training and testing respectively
hrir_size=ir.shape[3]; hrtf_size=ir.shape[3]//2+1 #get sizes for hrirs and hrtfs

Xf=np.fft.fft(ir,axis=3)[:,:,:,0:hrtf_size] #get HRTfs
Xm=np.apply_along_axis(minPhaseHRIR,3,Xf)
Xl=20*np.log10(np.abs(Xf)) #HRTFs in log magnitudes
Xl_mean=np.mean(Xl,axis=(0)); 


# read sofas files
sparse_itds=[]; sparse_pos=[]; sparse_ir=[]; sr=0; sparse_names=[]; sparse_sofa=[]
for file_path in sorted(glob.glob("LAP Task 2 Sparse HRTFs-selected/LAPtask2_*.sofa")):
    tmp=file_path.split("/")[1].split("_"); sparse_names.append([int(tmp[1]),int(tmp[2].split(".")[0])])
    sofa = sofar.read_sofa(file_path,verbose=0)
    tmp=np.lexsort((sofa.SourcePosition[:,0],sofa.SourcePosition[:,1])); 
    sofa.SourcePosition=sofa.SourcePosition[tmp,:]; sofa.Data_IR=sofa.Data_IR[tmp,:,:]
    sparse_pos.append(sofa.SourcePosition); sparse_ir.append(sofa.Data_IR); sr= sofa.Data_SamplingRate; 
    sparse_sofa.append(sofa)
    
n_measurements=np.unique(np.array(sparse_names)[:,0]);
n_pos=[None]*len(n_measurements);
for i in sparse_pos:
  n_pos[np.where(len(i)==n_measurements)[0][0]]=i

ord=np.unique(np.array(sparse_names)[:,0])
data=[[],[],[],[]]

for i in range(len(sparse_ir)):
    [_,tmp_itds,_]=itd_estimator_maxiacce(sparse_ir[i],48000); sparse_itds.append(tmp_itds)
    tmp_idx=np.array([False]*pos.shape[0])
    for j in n_pos[np.where(ord==sparse_names[i][0])[0][0]]:
        tmp_idx[np.where(np.all(j==pos,axis=1))[0]]=True    
    tmp_irs=20*np.log10(np.abs(np.fft.fft(sparse_ir[i],axis=2)))[:,:,0:129]
    freq= np.arange(256//2+1) *(48000 / 256); 
    tmp_irs=(tmp_irs-Xl_mean[tmp_idx,:,:]); tmp_irs[:,:,freq>20000]=0
    tmp_irs=back_from_mag(np.power(10,tmp_irs/20))
    tmp_irs[:,:,42:]=0; tmp_irs=tmp_irs[:,:,0:128]
    freq= np.arange(128//2+1) *(48000 / 128); 
    tmp_irs=20*np.log10(np.abs(np.fft.fft(tmp_irs,axis=2)))[:,:,0:sum(freq<=20000)]
    data[np.where(ord==sparse_names[i][0])[0][0]].append([tmp_irs,tmp_itds])

# %%
freq= np.arange(256//2+1) *(48000 / 256); 
Xp=(Xl-Xl_mean); Xp[:,:,:,freq>20000]=0
Xp=back_from_mag(np.power(10,Xp/20)); 
Xp[:,:,:,42:]=0; 
Xp=Xp[:,:,:,0:128]
freq= np.arange(128//2+1) *(48000 / 128); 
Xp=20*np.log10(np.abs(np.fft.fft(Xp,axis=3)))[:,:,:,0:sum(freq<=20000)]

# %%
#load previous models
all_models=[]
for i in range(4):
    tmp=[]
    for j in range(2):
        tmp1=[]
        if i==1 or i==3:
            for k in range(12):
                tmp1.append(keras.models.load_model("./Saved/"+str(i)+"_0_"+str(j)+"_"+str(k)))
        else:
            for k in range(11):
                tmp1.append(keras.models.load_model("./Saved/"+str(i)+"_0_"+str(j)+"_"+str(k)))
        tmp.append(tmp1)
    all_models.append([tmp,keras.models.load_model("./Saved/"+str(i)+"_1")])

# %% [markdown]
# # Training model

# %%
all_models=[]; 
for I0 in range(1,4):
    callback = keras.callbacks.EarlyStopping(monitor='val_loss',restore_best_weights=True, patience=120) #callback to abort after detecting overtraning
    inds=np.array([True]*ir.shape[1])
    for i in n_pos[I0]:
        inds[np.where(np.all(i==pos,axis=1))[0]]=False
    Xs=Xp/np.max(abs(Xp))
    train_l=np.append(Xs[idx,:,0,:],(Xs[idx,:,1,:])[:,flip_azis(pos),:],axis=0)
    test_l=np.append(Xs[~idx,:,0,:],(Xs[~idx,:,1,:])[:,flip_azis(pos),:],axis=0)
    x_train_l=np.zeros(train_l.shape); x_test_l=np.zeros(test_l.shape);
    eles=[90,75,60,45,30,20,10,0,-10,-20,-30,-45]
    models_l=[]; 
    for i,j in enumerate(eles):
        keras.backend.clear_session(); gc.collect()
        if i==0:
            if inds[-1]==False:
                continue
            else:
                [tmp,_,_]=build_model((sum(~inds),54),[["f",[30,None],[54,None]],[]])
                models_l.append(tmp); models_l[-1].compile(optimizer=keras.optimizers.Adam(amsgrad=True), loss=keras.losses.MeanSquaredError())
                models_l[-1].fit(train_l[:,~inds,:], train_l[:,pos[:,1]==j,:],epochs=1000,batch_size=int(len(train_l)*0.05),shuffle=True,validation_data=(test_l[:,~inds,:],test_l[:,pos[:,1]==j,:]),callbacks=[callback])
                x_train_l[:,pos[:,1]==j,:]=np.expand_dims(models_l[-1].predict(train_l[:,~inds,:]),axis=1);  x_test_l[:,pos[:,1]==j,:]=np.expand_dims(models_l[-1].predict(test_l[:,~inds,:]),axis=1);
        elif i==1:
            if inds[-1]==False:
                [tmp,_,_]=build_model((sum(~inds),54),[["f",[30,None],[54*72,None],(72,54)],[]])
                models_l.append(tmp); models_l[-1].compile(optimizer=keras.optimizers.Adam(amsgrad=True), loss=keras.losses.MeanSquaredError())
                models_l[-1].fit(train_l[:,~inds,:], train_l[:,pos[:,1]==j,:],epochs=1000,batch_size=int(len(train_l)*0.05),shuffle=True,validation_data=(test_l[:,~inds,:],test_l[:,pos[:,1]==j,:]),callbacks=[callback])
                x_train_l[:,pos[:,1]==j,:]=models_l[-1].predict(train_l[:,~inds,:]);  x_test_l[:,pos[:,1]==j,:]=models_l[-1].predict(test_l[:,~inds,:]); 
            else:
                [tmp,_,_]=build_model((sum(~inds)+1,54),[["f",[30,None],[54*72,None],(72,54)],[]])
                models_l.append(tmp); models_l[-1].compile(optimizer=keras.optimizers.Adam(amsgrad=True), loss=keras.losses.MeanSquaredError())            
                models_l[-1].fit(np.append(train_l[:,~inds,:],x_train_l[:,pos[:,1]==eles[i-1],:],axis=1), train_l[:,pos[:,1]==j,:],epochs=1000,batch_size=int(len(train_l)*0.05),shuffle=True,validation_data=(np.append(test_l[:,~inds,:],x_test_l[:,pos[:,1]==eles[i-1],:],axis=1),test_l[:,pos[:,1]==j,:]),callbacks=[callback])
                x_train_l[:,pos[:,1]==j,:]=models_l[-1].predict(np.append(train_l[:,~inds,:],x_train_l[:,pos[:,1]==eles[i-1],:],axis=1));  x_test_l[:,pos[:,1]==j,:]=models_l[-1].predict(np.append(test_l[:,~inds,:],x_test_l[:,pos[:,1]==eles[i-1],:],axis=1));             
        else:
            [tmp,_,_]=build_model((sum(~inds)+72,54),[["f",[30,None],[54*72,None],(72,54)],[]])
            models_l.append(tmp); models_l[-1].compile(optimizer=keras.optimizers.Adam(amsgrad=True), loss=keras.losses.MeanSquaredError())            
            models_l[-1].fit(np.append(train_l[:,~inds,:],x_train_l[:,pos[:,1]==eles[i-1],:],axis=1), train_l[:,pos[:,1]==j,:],epochs=1000,batch_size=int(len(train_l)*0.05),shuffle=True,validation_data=(np.append(test_l[:,~inds,:],x_test_l[:,pos[:,1]==eles[i-1],:],axis=1),test_l[:,pos[:,1]==j,:]),callbacks=[callback])
            x_train_l[:,pos[:,1]==j,:]=models_l[-1].predict(np.append(train_l[:,~inds,:],x_train_l[:,pos[:,1]==eles[i-1],:],axis=1));  x_test_l[:,pos[:,1]==j,:]=models_l[-1].predict(np.append(test_l[:,~inds,:],x_test_l[:,pos[:,1]==eles[i-1],:],axis=1));             
    x_train_l[:,pos[:,1]==90,:]=train_l[:,pos[:,1]==90,:]; x_test_l[:,pos[:,1]==90,:]=test_l[:,pos[:,1]==90,:]

    train_r=np.append(Xs[idx,:,1,:],(Xs[idx,:,0,:])[:,flip_azis(pos),:],axis=0)
    test_r=np.append(Xs[~idx,:,1,:],(Xs[~idx,:,0,:])[:,flip_azis(pos),:],axis=0)
    x_train_r=np.zeros(train_r.shape); x_test_r=np.zeros(test_r.shape);
    models_r=[]; 
    for i,j in enumerate(eles):
        keras.backend.clear_session(); gc.collect()
        if i==0:
            if inds[-1]==False:
                x_train_r[:,pos[:,1]==90,:]=train_r[:,pos[:,1]==90,:]; x_test_r[:,pos[:,1]==90,:]=test_r[:,pos[:,1]==90,:]
                continue
            else:
                [tmp,_,_]=build_model((sum(~inds),54),[["f",[30,None],[54,None]],[]])
                models_r.append(tmp); models_r[-1].compile(optimizer=keras.optimizers.Adam(amsgrad=True), loss=keras.losses.MeanSquaredError())
                models_r[-1].fit(train_r[:,~inds,:], train_r[:,pos[:,1]==j,:],epochs=1000,batch_size=int(len(train_r)*0.05),shuffle=True,validation_data=(test_r[:,~inds,:],test_r[:,pos[:,1]==j,:]),callbacks=[callback])
                x_train_r[:,pos[:,1]==j,:]=np.expand_dims(models_r[-1].predict(train_r[:,~inds,:]),axis=1);  x_test_r[:,pos[:,1]==j,:]=np.expand_dims(models_r[-1].predict(test_r[:,~inds,:]),axis=1);
        elif i==1:
            if inds[-1]==False:
                [tmp,_,_]=build_model((sum(~inds),54),[["f",[30,None],[54*72,None],(72,54)],[]])
                models_r.append(tmp); models_r[-1].compile(optimizer=keras.optimizers.Adam(amsgrad=True), loss=keras.losses.MeanSquaredError())
                models_r[-1].fit(train_r[:,~inds,:], train_r[:,pos[:,1]==j,:],epochs=1000,batch_size=int(len(train_r)*0.05),shuffle=True,validation_data=(test_r[:,~inds,:],test_r[:,pos[:,1]==j,:]),callbacks=[callback])
                x_train_r[:,pos[:,1]==j,:]=models_r[-1].predict(train_r[:,~inds,:]);  x_test_r[:,pos[:,1]==j,:]=models_r[-1].predict(test_r[:,~inds,:]); 
            else:
                [tmp,_,_]=build_model((sum(~inds)+1,54),[["f",[30,None],[54*72,None],(72,54)],[]])
                models_r.append(tmp); models_r[-1].compile(optimizer=keras.optimizers.Adam(amsgrad=True), loss=keras.losses.MeanSquaredError())            
                models_r[-1].fit(np.append(train_r[:,~inds,:],x_train_r[:,pos[:,1]==eles[i-1],:],axis=1), train_r[:,pos[:,1]==j,:],epochs=1000,batch_size=int(len(train_r)*0.05),shuffle=True,validation_data=(np.append(test_r[:,~inds,:],x_test_r[:,pos[:,1]==eles[i-1],:],axis=1),test_r[:,pos[:,1]==j,:]),callbacks=[callback])
                x_train_r[:,pos[:,1]==j,:]=models_r[-1].predict(np.append(train_r[:,~inds,:],x_train_r[:,pos[:,1]==eles[i-1],:],axis=1));  x_test_r[:,pos[:,1]==j,:]=models_r[-1].predict(np.append(test_r[:,~inds,:],x_test_r[:,pos[:,1]==eles[i-1],:],axis=1));             
        else:
            [tmp,_,_]=build_model((sum(~inds)+72,54),[["f",[30,None],[54*72,None],(72,54)],[]])
            models_r.append(tmp); models_r[-1].compile(optimizer=keras.optimizers.Adam(amsgrad=True), loss=keras.losses.MeanSquaredError())            
            models_r[-1].fit(np.append(train_r[:,~inds,:],x_train_r[:,pos[:,1]==eles[i-1],:],axis=1), train_r[:,pos[:,1]==j,:],epochs=1000,batch_size=int(len(train_r)*0.05),shuffle=True,validation_data=(np.append(test_r[:,~inds,:],x_test_r[:,pos[:,1]==eles[i-1],:],axis=1),test_r[:,pos[:,1]==j,:]),callbacks=[callback])
            x_train_r[:,pos[:,1]==j,:]=models_r[-1].predict(np.append(train_r[:,~inds,:],x_train_r[:,pos[:,1]==eles[i-1],:],axis=1));  x_test_r[:,pos[:,1]==j,:]=models_r[-1].predict(np.append(test_r[:,~inds,:],x_test_r[:,pos[:,1]==eles[i-1],:],axis=1));             

    max_itds=np.max(abs(itds))
    [itds_model,_,_]=build_model([sum(~inds)],[[[793,None]],[]])
    itds_model.compile(optimizer=keras.optimizers.Adam(amsgrad=True), loss=keras.losses.MeanSquaredError())
    itds_model.fit((itds[idx,:])[:,~inds], itds[idx,:],epochs=1000,batch_size=int(sum(idx)*0.05),shuffle=True,validation_data=((itds[~idx,:])[:,~inds],itds[~idx,:]),callbacks=[callback])
 
    # this save the models if required
    for i,j in enumerate(models_l):
      j.save("./Saved/"+str(I0)+"_0_0_"+str(i),save_format='tf')
    for i,j in enumerate(models_r):
      j.save("./Saved/"+str(I0)+"_0_1_"+str(i),save_format='tf')
    itds_model.save("./Saved/"+str(I0)+"_1",save_format='tf')
        
    all_models.append([[models_l,models_r],itds_model])

# %% [markdown]
# # Reconstruct from testing subject
# This requires to have the `FreeFieldCompMinPhase_48kHz` subjects in a folder called `SONICOM_F`

# %%
eles=[90,75,60,45,30,20,10,0,-10,-20,-30,-45]
results=[]
for I1 in range(4):
    for I0 in range(sum(~idx)):
    # for I0 in range(200):
        hrtf1=sofar.read_sofa("./SONICOM_F/"+(sofas[np.where(~idx)[0][I0]]).split("/")[1].split("_")[0]+"_FreeFieldCompMinPhase_48kHz.sofa")
        
        tmp=np.lexsort((hrtf1.SourcePosition[:,0],hrtf1.SourcePosition[:,1])); hrtf1.SourcePosition=hrtf1.SourcePosition[tmp,:] #sort by locations
        hrtf1.Data_IR=hrtf1.Data_IR[tmp,:,:]
        tmp_idx=np.array([False]*hrtf1.SourcePosition.shape[0])
        for i in n_pos[I1]:
            tmp_idx[np.where(np.all(i==hrtf1.SourcePosition,axis=1))[0]]=True
        inds=np.array([False]*ir.shape[1])
        for i in n_pos[I1]:
            inds[np.where(np.all(i==pos,axis=1))[0]]=True
        hrtf1.Data_IR=hrtf1.Data_IR[tmp_idx,:,:]
        # [_,tmp_itds,_]=hrtf_metrics.itd_estimator_maxiacce(hrtf1.Data_IR,48000) 
        [_,tmp_itds,_]=itd_estimator_maxiacce(hrtf1.Data_IR,48000)
        tmp_irs=20*np.log10(np.abs(np.fft.fft(hrtf1.Data_IR,axis=2)))[:,:,0:129]
        freq= np.arange(256//2+1) *(48000 / 256);
        tmp_irs=(tmp_irs-Xl_mean[inds,:,:]); tmp_irs[:,:,freq>20000]=0
        tmp_irs=back_from_mag(np.power(10,tmp_irs/20))
        tmp_irs[:,:,42:]=0; tmp_irs=tmp_irs[:,:,0:128]
        freq= np.arange(128//2+1) *(48000 / 128);
        tmp_irs=20*np.log10(np.abs(np.fft.fft(tmp_irs,axis=2)))[:,:,0:sum(freq<=20000)]/np.max(abs(Xp))

        x_data_l=np.zeros((Xp.shape[1],54))
        x_data_l[-1,:]=tmp_irs[-1,0,:];
        # for l,k in enumerate(all_models[I1][0][0]):
        for l,k in enumerate(eles):
            if l==0:
                if inds[-1]==True:
                    continue
                else:
                    x_data_l[pos[:,1]==eles[l],:]=np.expand_dims(all_models[I1][0][0][l].predict(tmp_irs[:,0,:].reshape((1,-1,54)),verbose=0),axis=1)
            elif l==1:
                if inds[-1]==True:
                    x_data_l[pos[:,1]==eles[l],:]=all_models[I1][0][0][l-1].predict(tmp_irs[:,0,:].reshape((1,-1,54)),verbose=0)[0,:,:]
                else:
                    x_data_l[pos[:,1]==eles[l],:]=all_models[I1][0][0][l+(-1*(inds[-1]==True))].predict(np.append(tmp_irs[:,0,:],x_data_l[pos[:,1]==eles[l-1],:],axis=0).reshape((1,-1,54)),verbose=0)[0,:,:]
            else:
                x_data_l[pos[:,1]==eles[l],:]=all_models[I1][0][0][l+(-1*(inds[-1]==True))].predict(np.append(tmp_irs[:,0,:],x_data_l[pos[:,1]==eles[l-1],:],axis=0).reshape((1,-1,54)),verbose=0)[0,:,:]
        x_data_r=np.zeros((Xp.shape[1],54))
        x_data_r[-1,:]=tmp_irs[-1,1,:]
        for l,k in enumerate(eles):
            if l==0:
                if inds[-1]==True:
                    continue
                else:
                    x_data_r[pos[:,1]==eles[l],:]=np.expand_dims(all_models[I1][0][1][l].predict(tmp_irs[:,1,:].reshape((1,-1,54)),verbose=0),axis=1)
            elif l==1:
                if inds[-1]==True:
                    x_data_r[pos[:,1]==eles[l],:]=all_models[I1][0][1][l-1].predict(tmp_irs[:,1,:].reshape((1,-1,54)),verbose=0)[0,:,:]
                else:
                    x_data_r[pos[:,1]==eles[l],:]=all_models[I1][0][1][l+(-1*(inds[-1]==True))].predict(np.append(tmp_irs[:,1,:],x_data_r[pos[:,1]==eles[l-1],:],axis=0).reshape((1,-1,54)),verbose=0)[0,:,:]
            else:
                x_data_r[pos[:,1]==eles[l],:]=all_models[I1][0][1][l+(-1*(inds[-1]==True))].predict(np.append(tmp_irs[:,1,:],x_data_r[pos[:,1]==eles[l-1],:],axis=0).reshape((1,-1,54)),verbose=0)[0,:,:]
        tmp_itds=all_models[I1][1].predict(tmp_itds.reshape((1,-1)),verbose=0)[0,:]
        x_data=np.copy(Xl_mean); 
        x_data_l=np.append(x_data_l,np.zeros((x_data_l.shape[0],65-54)),axis=1); x_data_r=np.append(x_data_r,np.zeros((x_data_l.shape[0],65-54)),axis=1);
        x_data_l=x_data_l*np.max(abs(Xp)); x_data_r=x_data_r*np.max(abs(Xp));
        x_data_l=back_from_mag(np.power(10,x_data_l/20)); x_data_r=back_from_mag(np.power(10,x_data_r/20)); 
        x_data_l=np.append(x_data_l,np.zeros((x_data_l.shape[0],256-128)),axis=1); x_data_r=np.append(x_data_r,np.zeros((x_data_l.shape[0],256-128)),axis=1);
        x_data_l=20*np.log10(abs(np.fft.fft(x_data_l)))[:,0:129]; x_data_r=20*np.log10(abs(np.fft.fft(x_data_r)))[:,0:129]
        x_data[:,0,:]=x_data[:,0,:]+x_data_l; x_data[:,1,:]=x_data[:,1,:]+x_data_r
        x_data=back_from_mag(np.power(10,x_data/20))
        for l in range(x_data.shape[0]):
            x_data[l,1 if tmp_itds[l]<0 else 0,:]=np.roll(x_data[l,1 if tmp_itds[l]<0 else 0,:],np.int32(np.rint(abs(tmp_itds[l]))))
        sofa = sofar.read_sofa("MyHRTF_FreeFieldCompMinPhase_48kHz.sofa",verbose=0)
        tmp=np.lexsort((sofa.SourcePosition[:,0],sofa.SourcePosition[:,1])); sofa.SourcePosition=sofa.SourcePosition[tmp,:]
        inds=np.array([False]*sofa.SourcePosition.shape[0])
        for l in sofa.SourcePosition:
            inds[np.where(np.all(l==pos,axis=1))[0]]=True
        sofa.Data_IR[inds,:,:]=x_data; sofa.Data_IR[sofa.SourcePosition[:,1]==90,:,:]=x_data[pos[:,1]==90,:,:]
        
        sofar.write_sofa("./tmp_Output/"+(sofas[np.where(~idx)[0][I0]]).split("/")[1].split("_")[0]+"_FreeFieldCompMinPhase_48kHz.sofa",sofa)

        hrtf1 = load_data.HRTF("./SONICOM_F/"+(sofas[np.where(~idx)[0][I0]]).split("/")[1].split("_")[0]+"_FreeFieldCompMinPhase_48kHz.sofa")
        hrtf2 = load_data.HRTF("./tmp_Output/"+(sofas[np.where(~idx)[0][I0]]).split("/")[1].split("_")[0]+"_FreeFieldCompMinPhase_48kHz.sofa")

        hrtf1,hrtf2 = load_data.match_hrtf_locations(hrtf1,hrtf2)
        itd_diff = hrtf_metrics.calculate_itd_difference(hrtf1,hrtf2)
        print(itd_diff)
        ild_diff = hrtf_metrics.calculate_ild_difference(hrtf1,hrtf2)
        print(ild_diff)
        lsd,lsd_mat = hrtf_metrics.calculate_lsd_across_locations(hrtf1.hrir,hrtf2.hrir,hrtf1.fs)
        print(lsd)
        results.append([I1,I0,itd_diff,ild_diff,lsd])

# %% [markdown]
# # Reconstruct challenge subject
# This requires to have the `FreeFieldCompMinPhase_48kHz` subjects in a folder called `SONICOM_F`

# %%
eles=[90,75,60,45,30,20,10,0,-10,-20,-30,-45]
for i,j in enumerate(data):
    for j3,j2 in enumerate(j):
        j2[0]=j2[0]/np.max(abs(Xp))
        x_data_l=np.zeros((Xp.shape[1],54))
        x_data_l[-1,:]=j2[0][-1,0,:]
        for l,k in enumerate(eles):
            if l==0:
                if np.isin(90,n_pos[i][:,1]):
                    continue
                else:
                    x_data_l[pos[:,1]==eles[l],:]=np.expand_dims(all_models[i][0][0][l].predict(j2[0][:,0,:].reshape((1,-1,54)),verbose=0),axis=1)
            elif l==1:
                if np.isin(90,n_pos[i][:,1]):
                    x_data_l[pos[:,1]==eles[l],:]=all_models[i][0][0][l-1].predict(j2[0][:,0,:].reshape((1,-1,54)),verbose=0)[0,:,:]
                else:
                    x_data_l[pos[:,1]==eles[l],:]=all_models[i][0][0][l+(-1*(np.isin(90,n_pos[i][:,1])))].predict(np.append(j2[0][:,0,:],x_data_l[pos[:,1]==eles[l-1],:],axis=0).reshape((1,-1,54)),verbose=0)[0,:,:]
            else:
                x_data_l[pos[:,1]==eles[l],:]=all_models[i][0][0][l+(-1*(np.isin(90,n_pos[i][:,1])))].predict(np.append(j2[0][:,0,:],x_data_l[pos[:,1]==eles[l-1],:],axis=0).reshape((1,-1,54)),verbose=0)[0,:,:]
        x_data_r=np.zeros((Xp.shape[1],54))
        x_data_r[-1,:]=j2[0][-1,1,:]
        for l,k in enumerate(eles):
            if l==0:
                if np.isin(90,n_pos[i][:,1]):
                    continue
                else:
                    x_data_r[pos[:,1]==eles[l],:]=np.expand_dims(all_models[i][0][1][l].predict(j2[0][:,1,:].reshape((1,-1,54)),verbose=0),axis=1)
            elif l==1:
                if np.isin(90,n_pos[i][:,1]):
                    x_data_r[pos[:,1]==eles[l],:]=all_models[i][0][1][l-1].predict(j2[0][:,1,:].reshape((1,-1,54)),verbose=0)[0,:,:]
                else:
                    x_data_r[pos[:,1]==eles[l],:]=all_models[i][0][1][l+(-1*(np.isin(90,n_pos[i][:,1])))].predict(np.append(j2[0][:,1,:],x_data_r[pos[:,1]==eles[l-1],:],axis=0).reshape((1,-1,54)),verbose=0)[0,:,:]
            else:
                x_data_r[pos[:,1]==eles[l],:]=all_models[i][0][1][l+(-1*(np.isin(90,n_pos[i][:,1])))].predict(np.append(j2[0][:,1,:],x_data_r[pos[:,1]==eles[l-1],:],axis=0).reshape((1,-1,54)),verbose=0)[0,:,:]
        tmp_itds=all_models[i][1].predict(j2[1].reshape((1,-1)),verbose=0)[0,:]
        x_data=np.copy(Xl_mean); 
        x_data_l=np.append(x_data_l,np.zeros((x_data_l.shape[0],65-54)),axis=1); x_data_r=np.append(x_data_r,np.zeros((x_data_l.shape[0],65-54)),axis=1);
        x_data_l=x_data_l*np.max(abs(Xp)); x_data_r=x_data_r*np.max(abs(Xp));
        x_data_l=back_from_mag(np.power(10,x_data_l/20)); x_data_r=back_from_mag(np.power(10,x_data_r/20)); 
        x_data_l=np.append(x_data_l,np.zeros((x_data_l.shape[0],256-128)),axis=1); x_data_r=np.append(x_data_r,np.zeros((x_data_l.shape[0],256-128)),axis=1);
        x_data_l=20*np.log10(abs(np.fft.fft(x_data_l)))[:,0:129]; x_data_r=20*np.log10(abs(np.fft.fft(x_data_r)))[:,0:129]
        x_data[:,0,:]=x_data[:,0,:]+x_data_l; x_data[:,1,:]=x_data[:,1,:]+x_data_r
        # x_data[:,1,:]=x_data[:,1,:]+x_data_l; x_data[:,0,:]=x_data[:,0,:]+x_data_r
        x_data=back_from_mag(np.power(10,x_data/20))
        for l in range(x_data.shape[0]):
            # x_data[l,0 if tmp_itds[l]<0 else 1,:]=np.roll(x_data[l,0 if tmp_itds[l]<0 else 1,:],np.int32(np.rint(abs(tmp_itds[l]))))
            x_data[l,1 if tmp_itds[l]<0 else 0,:]=np.roll(x_data[l,1 if tmp_itds[l]<0 else 0,:],np.int32(np.rint(abs(tmp_itds[l]))))
        #sofa.SourcePosition=sofa.SourcePosition[tmp,:]; sofa.Data_IR=sofa.Data_IR[tmp,:,:]
        sofa = sofar.read_sofa("MyHRTF_FreeFieldCompMinPhase_48kHz.sofa",verbose=0)
        tmp=np.lexsort((sofa.SourcePosition[:,0],sofa.SourcePosition[:,1])); sofa.SourcePosition=sofa.SourcePosition[tmp,:]
        inds=np.array([False]*sofa.SourcePosition.shape[0])
        for l in sofa.SourcePosition:
            inds[np.where(np.all(l==pos,axis=1))[0]]=True
        sofa.Data_IR[inds,:,:]=x_data; sofa.Data_IR[sofa.SourcePosition[:,1]==90,:,:]=x_data[pos[:,1]==90,:,:]
        sofar.write_sofa("./Output/LAPtask2_"+str(ord[i])+"_"+str(j3+1)+".sofa",sofa)


