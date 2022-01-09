# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 18:56:26 2021

@author: lmliang
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mne.time_frequency import tfr_morlet
from scipy.stats import zscore
#%% roadmap:
df = pd.read_excel("C:/Users/lmliang/Documents/GitHub/ishamProject/TR_EEGData_calc.xlsx")
#drop 14, 40, 59 delete 46 as well because trail number error
df = df[~df['Subject'].isin([14,40,46,59])]
#subid list
sublist = df['Subject'].unique()

#%% get reproduction performance.
encodingtime = df[df['Number'] == 2].reset_index()
reproductiontime = df[df['Number'] == 4].reset_index()
repRatio = df[df['Number'] == 2].reset_index()
repRatio['ratio'] = reproductiontime['Delay'].divide(encodingtime['Condition'])
#%% plot the reprpduction histogram by groups
g = sns.displot(reproductiontime[reproductiontime['Delay']<5000], x="Delay", col="Condition")
for ax, dur in zip(g.axes[0],[400,600,800,1600,1800,2000]):
    ax.axvline(dur, ls='--',color = 'r')
    
#%% plot the reprpduction ratio histogram by groups
g = sns.displot(repRatio[repRatio.ratio<5], x="ratio", col="Condition")
for ax, dur in zip(g.axes[0],[1,1,1,1,1,1]):
    ax.axvline(dur, ls='--',color = 'r')
    
#%% import EEG files, select [-2,4] time windows for encoding, and save as fif files
for subID in df['Subject'].unique():
    raw = mne.io.read_raw_eeglab('C:/Users/lmliang/Documents/MATLAB/ishamProject_data/TR_'+ str(subID) + '.set')
    # resampling
    raw  = raw.resample(sfreq = 250, n_jobs=-1)
        
    # high pass at 1 Hz, low pass at 50Hz, and apply a notch filter
    raw.filter(l_freq=1,h_freq = None)
    raw.filter(l_freq = None, h_freq = 50)
    raw.notch_filter(freqs=60)
    #average reference
    mne.set_eeg_reference(raw, ref_channels='average', copy=False, projection=False, ch_type='auto', verbose=None)
    
    # epoch
    events,event_id = mne.events_from_annotations(raw,event_id = {'400':4,'600':6,'800':8,'1600':16,'1800':18,'2000':20})
    epochs = mne.Epochs(raw, events, event_id, tmin=-2, tmax=4,baseline= None)

    epochs.save("C:/Users/lmliang/Documents/GitHub/ishamProject/mne_epoch_data/sub"+str(subID)+'-epo.fif', overwrite=True)

#%% calculate TFR
freqs =  2**np.linspace(1,5,num=15)
powerdf = pd.DataFrame()
powerMatrix = [[],[],[],[],[],[]]
#first import the epoch files
for subID in sublist:
    epochs = mne.read_epochs("C:/Users/lmliang/Documents/GitHub/ishamProject/mne_epoch_data/sub"+str(subID)+'-epo.fif')
    
    tfr  =  tfr_morlet(epochs, freqs = freqs, n_cycles = 6, n_jobs=-1, return_itc=False, average=False,use_fft = True, output='power',verbose='WARNING')
    
    tfr.info.set_montage(mne.channels.make_standard_montage('standard_1020'))
    
    #for subseconds, take 0-1s/  for supraseconds, take all durations
    for icond,tmax,ii in zip([400,600,800,1600,1800,2000],[1,1,1,1.6,1.8,2],[0,1,2,3,4,5]):
        test = tfr[str(icond)].crop(tmin=0,tmax=tmax).data.mean(-1)
        powerMatrix[ii].append(test)
        
#%%plot topoplots
avg_theta_power = np.array([zscore(np.log10(np.vstack([y[np.newaxis,...] for y in x])[:,:,:,4:8].mean((0,1,3)))) for x in powerMatrix])

fig, ax = plt.subplots(ncols=6, figsize=(16, 4), gridspec_kw=dict(top=0.9),
                       sharex=True, sharey=True)
for ii in range(6):
   im,cm = mne.viz.plot_topomap(avg_theta_power[ii], tfr.info,axes = ax[ii],show=False, contours = 0)
   
cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.9])
clb = fig.colorbar(im, cax=cbar_ax)

#%% plot psd
avg_psd = np.array([np.vstack([y[np.newaxis,...] for y in x])[:,:,1,:].mean((0,1)) for x in powerMatrix])

plt.style.use('default')
sns.set_style('white')
dum = []
for ii,cond in zip(range(6),[400,600,800,1600,1800,2000]):
    test = pd.DataFrame({'power':avg_psd[ii],'freq':freqs,'cond':cond})
    dum.append(test)
psd_df = pd.concat(dum,ignore_index=True)

cmap = plt.cm.get_cmap('tab20c')
palette = [cmap(2),cmap(1),cmap(0),cmap(6),cmap(5),cmap(4)]
fig,ax = plt.subplots()
sns.lineplot(data=psd_df, x="freq", y="power", hue="cond", ci = None,palette =palette)
plt.title('2-32Hz power spectra at Fz')

#%% import mental health score
mental_df = pd.read_excel("C:/Users/lmliang/Documents/GitHub/ishamProject/mentalHealth_forMing.xlsx")
#drop 14, 40, 59 delete 46 as well because trail number error
mental_df['subID'] = mental_df.SID.str.extract('(\d+)')
mental_df.subID = pd.to_numeric(mental_df.subID)
mental_df = mental_df.drop(['SID'],axis =1 )

mental_df = mental_df[mental_df.subID.isin(sublist)]


#%% calculate the association between mental health and reproduction
import statsmodels.formula.api as smf
beh_df = repRatio.groupby(['Subject','Condition']).mean().ratio.reset_index()
beh_df['BDI'] = mental_df.BDI.repeat(6).reset_index().BDI
beh_df['StateAnxiety'] = mental_df['State Anxiety Score'].repeat(6).reset_index()['State Anxiety Score']
beh_df['TraitAnxiety'] = mental_df['Trait Anxiety Score'].repeat(6).reset_index()['Trait Anxiety Score']

beh_df['Condition'] = beh_df.Condition/1000
# depression: BDI
est1 = smf.ols(formula='ratio ~ Condition + BDI', data=beh_df).fit()
est1.summary()

# State Anxiety Score	
est2 = smf.ols(formula='ratio ~ Condition + StateAnxiety', data=beh_df).fit()
est2.summary()

# Trait Anxiety Score
est3 = smf.ols(formula='ratio ~ Condition + TraitAnxiety', data=beh_df).fit()
est3.summary()

#%% 1. calculate the association between Fz theta power and reproduction duration
import statsmodels.formula.api as smf
subj_theta_Fz = np.array([zscore(np.log10(np.vstack([y[np.newaxis,...] for y in x])[:,:,1,4:8])).mean((-1,-2)) for x in powerMatrix])

for jj,subID in zip(range(50),sublist):
    for ii,cond in zip(range(6),[0.4,0.6,0.8,1.6,1.8,2.0]):
        beh_df.loc[ (beh_df['Subject'] == subID) & (beh_df['Condition'] == cond), 'thetaPower'] = subj_theta_Fz[ii,jj]
    
# regression Fz theta and reproduction duration
est4 = smf.ols(formula='ratio ~ Condition + thetaPower', data=beh_df).fit()
est4.summary()
#%%
plotMultiRegress('BDI',np.linspace(0, 26, 30),beh_df)
plotMultiRegress('StateAnxiety',np.linspace(20, 64, 30),beh_df)
plotMultiRegress('TraitAnxiety',np.linspace(20, 64, 30),beh_df)
plotMultiRegress('thetaPower',np.linspace(-2, 2, 30),beh_df)
#%% association between 1 and mental health
trialwise_theta_Fz = np.array([zscore(np.log10(np.vstack([y[np.newaxis,...] for y in x])[:,:,1,4:8])).mean(-1) for x in powerMatrix])

trial_df = repRatio.groupby(['Subject','Condition','Trial']).mean().ratio.reset_index()
trial_df['BDI'] = mental_df.BDI.repeat(60).reset_index().BDI
trial_df['StateAnxiety'] = mental_df['State Anxiety Score'].repeat(60).reset_index()['State Anxiety Score']
trial_df['TraitAnxiety'] = mental_df['Trait Anxiety Score'].repeat(60).reset_index()['Trait Anxiety Score']

trial_df['Condition'] = trial_df.Condition/1000

for jj,subID in zip(range(50),sublist):
    for ii,cond in zip(range(6),[0.4,0.6,0.8,1.6,1.8,2.0]):
        for itrial in range(10):
            trial_df.loc[ (trial_df['Subject'] == subID) & (trial_df['Condition'] == cond) & (trial_df['Trial'] == itrial+1), 'thetaPower'] = trialwise_theta_Fz[ii,jj,itrial]
        
#obtain subject level correlation Fz theta and reproduction duration

#%% first, with subsecond durations
sub_tvals = []
for subID in sublist:
    est = smf.ols(formula='ratio ~ Condition + thetaPower', data=trial_df.query('Subject =='+str(subID) +'and Condition < 1')).fit()
    sub_tvals.append(est.tvalues[2])

# now correlate the correlation values with mental health scores
corr_df = beh_df.groupby('Subject').mean().reset_index()
corr_df['tval']= sub_tvals
est5 = smf.ols(formula='tval ~ BDI', data=corr_df).fit()
est5.summary()
fig,ax= plt.subplots()
sns.regplot(x="BDI", y="tval", data=corr_df,color='#FF2700') #p 0.053

est6 = smf.ols(formula='tval ~ StateAnxiety', data=corr_df).fit()
est6.summary()
fig,ax= plt.subplots()
sns.regplot(x="StateAnxiety", y="tval", data=corr_df,color='#FF2700') #p 0.04

est7 = smf.ols(formula='tval ~ TraitAnxiety', data=corr_df).fit()
est7.summary() 
fig,ax= plt.subplots()
sns.regplot(x="TraitAnxiety", y="tval", data=corr_df,color='#FF2700') # p 0.97


#%% Second, with SUpra durations
sub_tvals = []
for subID in sublist:
    est = smf.ols(formula='ratio ~ Condition + thetaPower', data=trial_df.query('Subject =='+str(subID) +'and Condition > 1')).fit()
    sub_tvals.append(est.tvalues[2])

# now correlate the correlation values with mental health scores
corr_df = beh_df.groupby('Subject').mean().reset_index()
corr_df['tval']= sub_tvals
est8 = smf.ols(formula='tval ~ BDI', data=corr_df).fit()
est8.summary()
fig,ax= plt.subplots()
sns.regplot(x="BDI", y="tval", data=corr_df,color='#FF2700') #p 0.78

est9 = smf.ols(formula='tval ~ StateAnxiety', data=corr_df).fit()
est9.summary()
fig,ax= plt.subplots()
sns.regplot(x="StateAnxiety", y="tval", data=corr_df,color='#FF2700') #p 0.62

est10 = smf.ols(formula='tval ~ TraitAnxiety', data=corr_df).fit()
est10.summary()
fig,ax= plt.subplots()
sns.regplot(x="TraitAnxiety", y="tval", data=corr_df,color='#FF2700') # p 0.87

#%% plot multiple regression
# source https://medium.com/swlh/multi-linear-regression-using-python-44bd0d10082d
from sklearn import linear_model
def plotMultiRegress(yName, yy_pred, beh_df):
    # Prepare data
    X = beh_df[['Condition',yName]].values
    Y = beh_df['ratio'].values
    
    # Create range for each dimension
    x = X[:, 0]
    y = X[:, 1]
    z = Y
    
    xx_pred =  np.linspace(0, 2, 30)  # range of time values
    # yy_pred = np.linspace(0, 26, 30)  # range of BDI values
    xx_pred, yy_pred = np.meshgrid(xx_pred, yy_pred)
    model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T
    
    # Predict using model built on previous step
    regr = linear_model.LinearRegression()
    model = regr.fit(X, Y)
    predicted = model.predict(model_viz)
    
    # Evaluate model by using it's R^2 score 
    r2 = model.score(X, Y)
    
    # Plot model visualization
    plt.style.use('fivethirtyeight')
    
    fig = plt.figure(figsize=(12, 4))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    axes = [ax1, ax2, ax3]
    
    for ax in axes:
        ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
        ax.set_xlabel('Time[s]', fontsize=12)
        ax.set_ylabel(yName, fontsize=12)
        ax.set_zlabel('Ratio', fontsize=12)
        ax.locator_params(nbins=4, axis='x')
        ax.locator_params(nbins=5, axis='x')
    
    ax1.view_init(elev=25, azim=-60)
    ax2.view_init(elev=15, azim=15)
    ax3.view_init(elev=25, azim=60)
    
    fig.suptitle('Multi-Linear Regression Model Visualization ($R^2 = %.2f$)' % r2, fontsize=15, color='k')
    
    fig.tight_layout()

##########################################
#%% deal with results in occipital alpha, POz
#%%plot topoplots
avg_alpha_power = np.array([zscore(np.log10(np.vstack([y[np.newaxis,...] for y in x])[:,:,:,7:10].mean((0,1,3)))) for x in powerMatrix])

fig, ax = plt.subplots(ncols=6, figsize=(16, 4), gridspec_kw=dict(top=0.9),
                       sharex=True, sharey=True)
for ii in range(6):
   im,cm = mne.viz.plot_topomap(avg_alpha_power[ii], tfr.info,axes = ax[ii],show=False, contours = 0)
   
cbar_ax = fig.add_axes([0.01, 0.1, 0.02, 0.9])
clb = fig.colorbar(im, cax=cbar_ax)

#%% plot psd at electrode POz
avg_psd = np.array([np.vstack([y[np.newaxis,...] for y in x])[:,:,47,:].mean((0,1)) for x in powerMatrix])

dum = []
for ii,cond in zip(range(6),[400,600,800,1600,1800,2000]):
    test = pd.DataFrame({'power':avg_psd[ii],'freq':freqs,'cond':cond})
    dum.append(test)
psd_df = pd.concat(dum,ignore_index=True)

cmap = plt.cm.get_cmap('tab20c')
palette = [cmap(2),cmap(1),cmap(0),cmap(6),cmap(5),cmap(4)]
fig,ax = plt.subplots()
sns.lineplot(data=psd_df, x="freq", y="power", hue="cond", ci = None,palette =palette)
plt.title('2-32Hz power spectra at POz')

#%% import mental health score
mental_df = pd.read_excel("C:/Users/lmliang/Documents/GitHub/ishamProject/mentalHealth_forMing.xlsx")
#drop 14, 40, 59 delete 46 as well because trail number error
mental_df['subID'] = mental_df.SID.str.extract('(\d+)')
mental_df.subID = pd.to_numeric(mental_df.subID)
mental_df = mental_df.drop(['SID'],axis =1 )

mental_df = mental_df[mental_df.subID.isin(sublist)]


#%% 1. calculate the association between POz alpha power and reproduction duration
import statsmodels.formula.api as smf
subj_alpha_POz = np.array([zscore(np.log10(np.vstack([y[np.newaxis,...] for y in x])[:,:,47,7:10])).mean((-1,-2)) for x in powerMatrix])

for jj,subID in zip(range(50),sublist):
    for ii,cond in zip(range(6),[0.4,0.6,0.8,1.6,1.8,2.0]):
        beh_df.loc[ (beh_df['Subject'] == subID) & (beh_df['Condition'] == cond), 'alphaPower'] = subj_alpha_POz[ii,jj]
    
# regression POz alpha and reproduction duration
est4 = smf.ols(formula='ratio ~ Condition + alphaPower', data=beh_df).fit()
est4.summary()

plotMultiRegress('alphaPower',np.linspace(-3, 3, 30),beh_df)
#%% association between 1 and mental health
trialwise_alpha_POz = np.array([zscore(np.log10(np.vstack([y[np.newaxis,...] for y in x])[:,:,47,7:10])).mean(-1) for x in powerMatrix])

trial_df = repRatio.groupby(['Subject','Condition','Trial']).mean().ratio.reset_index()
trial_df['BDI'] = mental_df.BDI.repeat(60).reset_index().BDI
trial_df['StateAnxiety'] = mental_df['State Anxiety Score'].repeat(60).reset_index()['State Anxiety Score']
trial_df['TraitAnxiety'] = mental_df['Trait Anxiety Score'].repeat(60).reset_index()['Trait Anxiety Score']

trial_df['Condition'] = trial_df.Condition/1000

for jj,subID in zip(range(50),sublist):
    for ii,cond in zip(range(6),[0.4,0.6,0.8,1.6,1.8,2.0]):
        for itrial in range(10):
            trial_df.loc[ (trial_df['Subject'] == subID) & (trial_df['Condition'] == cond) & (trial_df['Trial'] == itrial+1), 'alphaPower'] = trialwise_alpha_POz[ii,jj,itrial]
        
#obtain subject level correlation Fz theta and reproduction duration

#%% first, with subsecond durations
sub_tvals = []
for subID in sublist:
    est = smf.ols(formula='ratio ~ Condition + alphaPower', data=trial_df.query('Subject =='+str(subID) +'and Condition < 1')).fit()
    sub_tvals.append(est.tvalues[2])

# now correlate the correlation values with mental health scores
corr_df = beh_df.groupby('Subject').mean().reset_index()
corr_df['tval']= sub_tvals
est11 = smf.ols(formula='tval ~ BDI', data=corr_df).fit()
est11.summary()
fig,ax= plt.subplots()
sns.regplot(x="BDI", y="tval", data=corr_df) #p 0.47

est12 = smf.ols(formula='tval ~ StateAnxiety', data=corr_df).fit()
est12.summary()
fig,ax= plt.subplots()
sns.regplot(x="StateAnxiety", y="tval", data=corr_df) #p 0.30

est13 = smf.ols(formula='tval ~ TraitAnxiety', data=corr_df).fit()
est13.summary()
fig,ax= plt.subplots()
sns.regplot(x="TraitAnxiety", y="tval", data=corr_df) # p 0.30


#%% Second, with SUpra durations
sub_tvals = []
for subID in sublist:
    est = smf.ols(formula='ratio ~ Condition + alphaPower', data=trial_df.query('Subject =='+str(subID) +'and Condition > 1')).fit()
    sub_tvals.append(est.tvalues[2])

# now correlate the correlation values with mental health scores
corr_df = beh_df.groupby('Subject').mean().reset_index()
corr_df['tval']= sub_tvals
est14 = smf.ols(formula='tval ~ BDI', data=corr_df).fit()
est14.summary()
fig,ax= plt.subplots()
sns.regplot(x="BDI", y="tval", data=corr_df) #p 0.051

est15 = smf.ols(formula='tval ~ StateAnxiety', data=corr_df).fit()
est15.summary()
fig,ax= plt.subplots()
sns.regplot(x="StateAnxiety", y="tval", data=corr_df) #p 0.86

est16 = smf.ols(formula='tval ~ TraitAnxiety', data=corr_df).fit()
est16.summary()
fig,ax= plt.subplots()
sns.regplot(x="TraitAnxiety", y="tval", data=corr_df) # p 0.024
