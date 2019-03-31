from numpy.random import seed
seed(1017)

import os
from glob import glob
from collections import OrderedDict

import mne
from mne.io import RawArray
from mne import read_evokeds, read_source_spaces, compute_covariance
from mne import channels, find_events, concatenate_raws
from mne import pick_types, viz, io, Epochs, create_info
from mne import pick_channels, concatenate_epochs
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_raw
from mne.channels import read_montage
from mne.time_frequency import tfr_morlet

import numpy as np
from numpy import genfromtxt

import pandas as pd
pd.options.display.precision = 4
pd.options.display.max_columns = None

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,12)


def LoadBVData(sub,session,data_dir,exp):
  #for isub,sub in enumerate(subs):       
  print('Loading data for subject number: ' + sub)
  fname = data_dir + exp + '/' + sub + '_' + exp + '_' + session + '.vhdr'
  raw,sfreq = loadBV(fname,plot_sensors=False,plot_raw=False,
          plot_raw_psd=False,stim_channel=True)
  return raw

def loadBV(filename, plot_sensors=True, plot_raw=True,
  plot_raw_psd=True, stim_channel=False, ):
  """Load in recorder data files."""


  #load .vhdr files from brain vision recorder
  raw = io.read_raw_brainvision(filename,
                          montage='standard_1020',
                          eog=('HEOG', 'VEOG'),
                          preload=True,stim_channel=stim_channel)

  #set sampling rate
  sfreq = raw.info['sfreq']
  print('Sampling Rate = ' + str(sfreq))

  #load channel locations
  print('Loading Channel Locations')
  if plot_sensors:
    raw.plot_sensors(show_names='True')

  ##Plot raw data
  if plot_raw:
    raw.plot(n_channels=16, block=True)

   #plot raw psd
  if plot_raw_psd:
    raw.plot_psd(fmin=.1, fmax=100 )

  return raw, sfreq


def LoadMuseData(subs, nsesh, data_dir, load_verbose=False, sfreq=256.):
  nsubs = len(subs)
  raw = []
  print('Loading Data')
  for isub,sub in enumerate(subs):
    print('Subject number ' + str(isub+1) + '/' + str(nsubs))
    for isesh in range(nsesh):
      print(' Session number ' + str(isesh+1) + '/' + str(nsesh))
      raw.append(muse_load_data(data_dir, sfreq=sfreq ,subject_nb=sub,
                    session_nb=isesh+1,verbose=load_verbose))
  raw = concatenate_raws(raw)
  return raw


#from eeg-notebooks load_data
def muse_load_data(data_dir, subject_nb=1, session_nb=1, sfreq=256.,
                   ch_ind=[0, 1, 2, 3], stim_ind=5, replace_ch_names=None,
                   verbose=1):
    """Load CSV files from the /data directory into a Raw object.

    Args:
        data_dir (str): directory inside /data that contains the
            CSV files to load, e.g., 'auditory/P300'

    Keyword Args:
        subject_nb (int or str): subject number. If 'all', load all
            subjects.
        session_nb (int or str): session number. If 'all', load all
            sessions.
        sfreq (float): EEG sampling frequency
        ch_ind (list): indices of the EEG channels to keep
        stim_ind (int): index of the stim channel
        replace_ch_names (dict or None): dictionary containing a mapping to
            rename channels. Useful when an external electrode was used.

    Returns:
        (mne.io.array.array.RawArray): loaded EEG
    """


    if subject_nb == 'all':
        subject_nb = '*'
    if session_nb == 'all':
        session_nb = '*'

    data_path = os.path.join(
            'eeg-notebooks/data', data_dir,
            'subject{}/session{}/*.csv'.format(subject_nb, session_nb))
    fnames = glob(data_path)

    return load_muse_csv_as_raw(fnames,
                                sfreq=sfreq,
                                ch_ind=ch_ind,
                                stim_ind=stim_ind,
                                replace_ch_names=replace_ch_names,
                                verbose=verbose)


#from eeg-notebooks
def load_muse_csv_as_raw(filename, sfreq=256., ch_ind=[0, 1, 2, 3],
                         stim_ind=5, replace_ch_names=None, verbose=1):
    """Load CSV files into a Raw object.

    Args:
        filename (str or list): path or paths to CSV files to load

    Keyword Args:
        subject_nb (int or str): subject number. If 'all', load all
            subjects.
        session_nb (int or str): session number. If 'all', load all
            sessions.
        sfreq (float): EEG sampling frequency
        ch_ind (list): indices of the EEG channels to keep
        stim_ind (int): index of the stim channel
        replace_ch_names (dict or None): dictionary containing a mapping to
            rename channels. Useful when an external electrode was used.

    Returns:
        (mne.io.array.array.RawArray): loaded EEG
    """

    n_channel = len(ch_ind)

    raw = []
    for fname in filename:
        # read the file
        data = pd.read_csv(fname, index_col=0)

        # name of each channels
        ch_names = list(data.columns)[0:n_channel] + ['Stim']

        if replace_ch_names is not None:
            ch_names = [c if c not in replace_ch_names.keys()
                        else replace_ch_names[c] for c in ch_names]

        # type of each channels
        ch_types = ['eeg'] * n_channel + ['stim']
        montage = read_montage('standard_1005')

        # get data and exclude Aux channel
        data = data.values[:, ch_ind + [stim_ind]].T

        # convert in Volts (from uVolts)
        data[:-1] *= 1e-6

        # create MNE object
        info = create_info(ch_names=ch_names, ch_types=ch_types,
                           sfreq=sfreq, montage=montage, verbose=verbose)
        raw.append(RawArray(data=data, info=info, verbose=verbose))

    # concatenate all raw objects
    if len(raw) > 0:
      raws = concatenate_raws(raw, verbose=verbose)
    else:
      print('No files for subject with filename ' + str(filename))
      raws = raw
      
    return raws


def SimulateRaw(amp1 = 50, amp2 = 100, freq = 1., batch=1):

  """Create simulated raw data and events of two kinds
  
  Keyword Args:
      amp1 (float): amplitude of first condition effect
      amp2 (float): ampltiude of second condition effect, 
          null hypothesis amp1=amp2
      freq (float): Frequency of simulated signal 1. for ERP 10. for alpha
      batch (int): number of groups of 255 trials in each condition
  Returns: 
      raw: simulated EEG MNE raw object with two event types
      event_id: dict of the two events for input to PreProcess()
  """


  data_path = sample.data_path()
  raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
  trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
  src_fname = data_path + '/subjects/sample/bem/sample-oct-6-src.fif'
  bem_fname = (data_path + 
        '/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif')

  
  raw_single = mne.io.read_raw_fif(raw_fname,preload=True)
  raw_single.set_eeg_reference(projection=True)
  raw_single = raw_single.crop(0., 255.)
  raw_single = raw_single.copy().pick_types(meg=False, eeg=True, eog=True, stim=True)

  #concatenate 4 raws together to make 1000 trials
  raw = []
  for i in range(batch):
    raw.append(raw_single)
  raw = concatenate_raws(raw)

  epoch_duration = 1.
  
  def data_fun(amp, freq):
    """Create function to create fake signal"""
    def data_fun_inner(times):
      """Create fake signal with no noise"""
      n_samp = len(times)
      window = np.zeros(n_samp)
      start, stop = [int(ii * float(n_samp) / 2)
        for ii in (0, 1)]
      window[start:stop] = np.hamming(stop - start)
      data = amp * 1e-9 * np.sin(2. * np.pi * freq * times)
      data *= window
      return data
    return data_fun_inner

  times = raw.times[:int(raw.info['sfreq'] * epoch_duration)]
  src = read_source_spaces(src_fname)

  stc_zero = simulate_sparse_stc(src, n_dipoles=1, times=times,
              data_fun=data_fun(amp1,freq), random_state=0)
  stc_one = simulate_sparse_stc(src, n_dipoles=1, times=times,
              data_fun=data_fun(amp2,freq), random_state=0)

  raw_sim_zero = simulate_raw(raw, stc_zero, trans_fname, src, bem_fname, 
            cov='simple', blink=True, n_jobs=1, verbose=True)
  raw_sim_one = simulate_raw(raw, stc_one, trans_fname, src, bem_fname, 
            cov='simple', blink=True, n_jobs=1, verbose=True)

  stim_pick = raw_sim_one.info['ch_names'].index('STI 014')
  raw_sim_one._data[stim_pick][np.where(raw_sim_one._data[stim_pick]==1)] = 2
  raw = concatenate_raws([raw_sim_zero, raw_sim_one])
  event_id = {'CondZero': 1,'CondOne': 2}
  return raw, event_id


def mastoidReref(raw):
  ref_idx = pick_channels(raw.info['ch_names'],['M2'])
  eeg_idx = pick_types(raw.info,eeg=True)
  raw._data[eeg_idx,:] =  raw._data[eeg_idx,:]  -  raw._data[ref_idx,:] * .5 ;
  return raw

def GrattonEmcpRaw(raw):
  raw_eeg = raw.copy().pick_types(eeg=True)[:][0]
  raw_eog = raw.copy().pick_types(eog=True)[:][0]
  b = np.linalg.solve(np.dot(raw_eog,raw_eog.T), np.dot(raw_eog,raw_eeg.T))
  eeg_corrected = (raw_eeg.T - np.dot(raw_eog.T,b)).T
  raw_new = raw.copy()
  raw_new._data[pick_types(raw.info,eeg=True),:] = eeg_corrected
  return raw_new


def GrattonEmcpEpochs(epochs):
  '''
  # Correct EEG data for EOG artifacts with regression
  # INPUT - MNE epochs object (with eeg and eog channels)
  # OUTPUT - MNE epochs object (with eeg corrected)
  # After: Gratton,Coles,Donchin, 1983
  # -compute the ERP in each condition
  # -subtract ERP from each trial
  # -subtract baseline (mean over all epoch)
  # -predict eye channel remainder from eeg remainder
  # -use coefficients to subtract eog from eeg
  '''

  event_names = ['A_error','B_error']
  i = 0
  for key, value in sorted(epochs.event_id.items(), key=lambda x: (x[1], x[0])):
    event_names[i] = key
    i += 1

  #select the correct channels and data
  eeg_chans = pick_types(epochs.info, eeg=True, eog=False)
  eog_chans = pick_types(epochs.info, eeg=False, eog=True)
  original_data = epochs._data

  #subtract the average over trials from each trial
  rem = {}
  for event in event_names:
    data = epochs[event]._data
    avg = np.mean(epochs[event]._data,axis=0)
    rem[event] = data-avg

  #concatenate trials together of different types
  ## then put them all back together in X (regression on all at once)
  allrem = np.concatenate([rem[event] for event in event_names])

  #separate eog and eeg
  X = allrem[:,eeg_chans,:]
  Y = allrem[:,eog_chans,:]

  #subtract mean over time from every trial/channel
  X = (X.T - np.mean(X,2).T).T
  Y = (Y.T - np.mean(Y,2).T).T

  #move electrodes first
  X = np.moveaxis(X,0,1)
  Y = np.moveaxis(Y,0,1)

  #make 2d and compute regression
  X = np.reshape(X,(X.shape[0],np.prod(X.shape[1:])))
  Y = np.reshape(Y,(Y.shape[0],np.prod(Y.shape[1:])))
  b = np.linalg.solve(np.dot(Y,Y.T), np.dot(Y,X.T))

  #get original data and electrodes first for matrix math
  raw_eeg = np.moveaxis(original_data[:,eeg_chans,:],0,1)
  raw_eog = np.moveaxis(original_data[:,eog_chans,:],0,1)

  #subtract weighted eye channels from eeg channels
  eeg_corrected = (raw_eeg.T - np.dot(raw_eog.T,b)).T

  #move back to match epochs
  eeg_corrected = np.moveaxis(eeg_corrected,0,1)

  #copy original epochs and replace with corrected data
  epochs_new = epochs.copy()
  epochs_new._data[:,eeg_chans,:] = eeg_corrected

  return epochs_new


def PreProcess(raw, event_id, plot_psd=False, filter_data=True,
               filter_range=(1,30), plot_events=False, epoch_time=(-.2,1),
               baseline=(-.2,0), rej_thresh_uV=200, rereference=False, 
               emcp_raw=False, emcp_epochs=False, epoch_decim=1, plot_electrodes=False,
               plot_erp=False):

  sfreq = raw.info['sfreq']
  #create new output freq for after epoch or wavelet decim
  nsfreq = sfreq/epoch_decim
  tmin=epoch_time[0]
  tmax=epoch_time[1]
  if filter_range[1] > nsfreq:
    filter_range[1] = nsfreq/2.5  #lower than 2 to avoid aliasing from decim??

  #pull event names in order of trigger number
  event_names = ['A_error','B_error']
  i = 0
  for key, value in sorted(event_id.items(), key=lambda x: (x[1], x[0])):
    event_names[i] = key
    i += 1

  #Filtering
  if rereference:
    print('Rerefering to average mastoid')
    raw = mastoidReref(raw)

  if filter_data:
    print('Filtering Data Between ' + str(filter_range[0]) + 
            ' and ' + str(filter_range[1]) + ' Hz.')
    raw.filter(filter_range[0],filter_range[1],
               method='iir', verbose='WARNING' )

  if plot_psd:
    raw.plot_psd(fmin=filter_range[0], fmax=nsfreq/2 )

  #Eye Correction
  if emcp_raw:
    print('Raw Eye Movement Correction')
    raw = GrattonEmcpRaw(raw)

  #Epoching
  events = find_events(raw,shortest_event=1)
  color = {1: 'red', 2: 'black'}
  #artifact rejection
  rej_thresh = rej_thresh_uV*1e-6

  #plot event timing
  if plot_events:
    viz.plot_events(events, sfreq, raw.first_samp, color=color,
                        event_id=event_id)

  #Constructevents
  epochs = Epochs(raw, events=events, event_id=event_id,
                  tmin=tmin, tmax=tmax, baseline=baseline,
                  preload=True,reject={'eeg':rej_thresh},
                  verbose=False, decim=epoch_decim)
  print('Remaining Trials: ' + str(len(epochs)))

  if emcp_epochs:
    print('Epochs Eye Movement Correct')
    epochs = GrattonEmcpEpochs(epochs)

  evoked_dict = {event_names[0]:epochs[event_names[0]].average(),
                              event_names[1]:epochs[event_names[1]].average()}

  ## plot ERP at each electrode
  if plot_electrodes:
    picks = pick_types(evoked_dict[event_names[0]].info, meg=False, eeg=True, eog=False)
    fig_zero = evoked_dict[event_names[0]].plot(spatial_colors=True,picks=picks)
    fig_zero = evoked_dict[event_names[1]].plot(spatial_colors=True,picks=picks)

  ## plot ERP in each condition on same plot
  if plot_erp:
    #find the electrode most miximal on the head (highest in z)
    picks = np.argmax([evoked_dict[event_names[0]].info['chs'][i]['loc'][2] 
              for i in range(len(evoked_dict[event_names[0]].info['chs']))])
    colors = {event_names[0]:"Red",event_names[1]:"Blue"}
    viz.plot_compare_evokeds(evoked_dict,colors=colors,
                            picks=picks,split_legend=True)

  return epochs