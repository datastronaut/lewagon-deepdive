# IMPORTS

import librosa
import librosa.display
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import Audio
import random
from scipy import stats
from numpy import genfromtxt

# AUDIO DATA AUGMENTATION FUNCTIONC: WHITE NOISE AND RANDOM GAIN 

def add_white_noise(signal,noise_factor):
  noise = np.random.normal(0, signal.std(),signal.size)
  augmented_signal = signal + noise * noise_factor
  return augmented_signal

def random_gain(signal, min_gain_factor, max_gain_factor):
  gain_factor= random.uniform(min_gain_factor,max_gain_factor)
  return signal * gain_factor


# TRAIN - VALIDATION - TEST SPLIT TO PERFORM BEFOREHAND 

def dataset_split(audio_code_list, test_size=0.2):
  '''
  Performs a Train - Validation - Test split on the dataset (i.e. the audio_code_list)

  Inputs
  audio_code_list: list of tuples of signal (the features) and code (the targets)

  Outputs
  train, validation and test set as list of tuples of signal and code
  '''
  signals = []
  codes   = []

  for ad, c in audio_code_list:
    signals.append(ad)
    codes.append(c)

  X_train, X_test, y_train, y_test = train_test_split(signals,codes,test_size=test_size)
  X_train, X_val, y_train, y_val   = train_test_split(X_train,y_train,test_size=test_size)

  train_set=[]
  val_set  =[]
  test_set =[]

  for ad,c in zip(X_train,y_train):
      train_set.append((ad,c))

  for ad,c in zip(X_val,y_val):
      val_set.append((ad,c))

  for ad,c in zip(X_test,y_test):
      test_set.append((ad,c))

  return train_set,val_set, test_set

def enhanced_dataset_split(input_list,df, test_size=0.2, threshold=0.5, target='family'):
  '''
  Performs an enhanced dataset_split so as the train_set has got at least threshold*100 % of all
  species or families  

  Inputs
  input_list: list of tuples of signal and code
  df        : DataFrame with specific info, defined with get_dataset()
  test_size : test_size required by sklearn train_test_split function
  threshold : determines the class distribution in the train set
  target    : determines what the model should predict, species or family code

  Outputs
  train, validation and test set as list of tuples of signal and code
  '''
  if target == 'family': 
    condition = False
    while condition == False:
      train_set,val_set, test_set = dataset_split(input_list, test_size=test_size) # train val test split
      
      X_train_temp=[]
      y_train_temp=[]

      for x,y in train_set:
        X_train_temp.append(x)
        y_train_temp.append(y)

      # create 2 temporary datasets to compare families distribution in the train_set and the global set

      df_total= pd.DataFrame(df[['family_code']].value_counts()*threshold).rename(columns={0:'total_count'})
      df_temp=pd.DataFrame(pd.DataFrame(y_train_temp).value_counts()).rename(columns={0:'count'})
      df_temp['total_count']=df_total['total_count']
      df_temp['comparison']=np.where(df_temp['count'] >= df_temp['total_count'], 'True', 'False')

      if df_temp[df_temp['comparison'] == 'True'].count()['comparison'] == df_temp.shape[0]:
        condition = True

  elif target == 'species':
    condition = False
    while condition == False:
      train_set,val_set, test_set = dataset_split(input_list, test_size=test_size) # train val test split
      
      X_train_temp=[]
      y_train_temp=[]

      for x,y in train_set:
        X_train_temp.append(x)
        y_train_temp.append(y)

      # create 2 temporary datasets to compare species distribution in the train_set and the global set

      df_total= pd.DataFrame(df[['species_code']].value_counts()*threshold).rename(columns={0:'total_count'})
      df_temp=pd.DataFrame(pd.DataFrame(y_train_temp).value_counts()).rename(columns={0:'count'})
      df_temp['total_count']=df_total['total_count']
      df_temp['comparison']=np.where(df_temp['count'] >= df_temp['total_count'], 'True', 'False')

      if df_temp[df_temp['comparison'] == 'True'].count()['comparison'] == df_temp.shape[0]:
        condition = True

  return train_set,val_set, test_set

# SPLIT AUDIO FILES THAT ARE ABOVE AND BELOW TARGET_TIME 

def split_above_below(audio_code_list, target_time,sr):
  '''
  Splits the different datasets (train, val and test) whether the signals are above or below the target_time

  Inputs
  audio_code_list: list of tuples of signal and code
  target_time    : duration in seconds wanted for the signal
  sr             : sampling rate

  Outputs
  below: list of tuples of signal and code below or equal to target_time
  above: list of tuples of signal and code above target_time

  '''
  below = []
  above = []

  for ad, c in audio_code_list:
    if (len(ad) / sr) < target_time:
      below.append((ad,c))
    else:
      above.append((ad,c))

  return below,above

# PREPROCESSING SAMPLES THAT ARE ABOVE TAGET_TIME 

def train_split_above_samples(above,over_r,under_r, target_time, sr,audio_manipulation=False):
  '''
  Performs preprocessing actions on the train_set containing signals above the target_time

  Inputs
  above             : list of tuples of signal and code above target_time
  over_r            : list of code names of over_represented classes
  under_r           : list of code names of under_represented classes
  target_time       : duration in seconds wanted for the signal
  sr                : sampling rate
  audio_manipulation: if True, applies audio data augmentation on under represented samples

  Ouputs
  above_record_samples: list of tuples of signal and code of duration target_time,
                  with different preprocessing based on class representation
  '''
  under_r_samples = []
  over_r_samples  = []

  for signal, code in above:
    if code in under_r: # preprocessing actions for under-represented classes
      if audio_manipulation:
        # cut into target_time seconds consecutive slices, 3 times at different intervals 
        # for each slice, apply both the white noise and the random gain functions
        nb_split_samples = signal.size // (target_time*sr)

        for i in range(0,nb_split_samples):
          sound_1 = signal[i*(target_time*sr): (i+1)*(target_time*sr)]
          under_r_samples.append((sound_1,code))
          under_r_samples.append((add_white_noise(sound_1,0.1),code))
          under_r_samples.append((random_gain(sound_1,2,4),code))

          random_shift = random.randint(0,((target_time*sr)/3)*2)
          if len(signal) > ((i+1)*(target_time*sr))+random_shift:
            sound_2 = signal[((i*(target_time*sr))+random_shift): (((i+1)*(target_time*sr))+random_shift)]
            under_r_samples.append((sound_2,code))
            under_r_samples.append((add_white_noise(sound_2,0.1),code))
            under_r_samples.append((random_gain(sound_2,2,4),code))
          
          random_shift = random.randint(0,((target_time*sr)/3)*2)
          if len(signal) > ((i+1)*(target_time*sr))+random_shift:
            sound_3 = signal[((i*(target_time*sr))+random_shift): (((i+1)*(target_time*sr))+random_shift)]
            under_r_samples.append((sound_3,code))
            under_r_samples.append((add_white_noise(sound_3,0.1),code))
            under_r_samples.append((random_gain(sound_3,2,4),code))
        
      else:
        # cut into target_time seconds consecutive slices, 3 times at different intervals
        # no audio data augmentation
        nb_split_samples = signal.size // (target_time*sr)

        for i in range(0,nb_split_samples):
          sound_1 = signal[i*(target_time*sr): (i+1)*(target_time*sr)]
          under_r_samples.append((sound_1,code))

          random_shift = random.randint(0,((target_time*sr)/3)*2)
          if len(signal) > ((i+1)*(target_time*sr))+random_shift:
            sound_2 = signal[((i*(target_time*sr))+random_shift): (((i+1)*(target_time*sr))+random_shift)]
            under_r_samples.append((sound_2,code))
          
          random_shift = random.randint(0,((target_time*sr)/3)*2)
          if len(signal) > ((i+1)*(target_time*sr))+random_shift:
            sound_3 = signal[((i*(target_time*sr))+random_shift): (((i+1)*(target_time*sr))+random_shift)]
            under_r_samples.append((sound_3,code))
              
    else:  # preprocessing actions for overer-represented classes 
        # cut into target_time seconds consecutive slices + pad last slice if it is >= target_time - 1
        nb_split_samples = len(signal) // (target_time*sr) 

        for i in range(0,nb_split_samples):
          sound_i = signal[i*(target_time*sr): (i+1)*(target_time*sr)]
          over_r_samples.append((sound_i, code))

          if (len(signal) % (target_time*sr)) / sr >= target_time - 1 :
            len_to_pad=(sr*target_time) - len(signal[(nb_split_samples) * (target_time * sr):])
            a = random.randint(0, len_to_pad)
            b = len_to_pad - a
            audio = np.pad(signal[(nb_split_samples)*(target_time*sr):], (a,b), "constant")
            over_r_samples.append((audio, code))

  above_record_samples  = under_r_samples + over_r_samples

  return above_record_samples

# PREPROCESSING SAMPLES THAT ARE BELOW TARGET_TIME 

def train_split_below_samples(below,over_r,under_r, target_time, sr,audio_manipulation=False):
  '''
  Performs preprocessing actions on the train_set containing signals below the target_time

  Inputs
  below             : list of tuples of signal and code below or equal to target_time
  over_r            : list of code names of over_represented classes
  under_r           : list of code names of under_represented classes
  target_time       : duration in seconds wanted for the signal
  sr                : sampling rate
  audio_manipulation: if True, applies audio data augmentation on under represented samples

  Ouputs
  below_record_samples: list of tuples of signal padded randomly and code 
  '''
  under_r_samples = []
  over_r_samples  = []

  for signal, code in below:
    if code in under_r: # preprocessing actions for under-represented classes
      if audio_manipulation:
      # 3 random pads per signal
      # for each signal, apply both the white noise and the random gain functions
        for i in range(3):  
            len_to_pad=(sr*target_time) - len(signal)
            a = random.randint(0, len_to_pad)
            b = len_to_pad - a
            audio = np.pad(signal, (a,b), "constant")
            under_r_samples.append((audio,code))
            under_r_samples.append((add_white_noise(audio,0.1),code))
            under_r_samples.append((random_gain(audio,2,4),code))
      
      else:
        # 3 random pads per signal
        # no audio data augmentation
        for i in range(3):  
            len_to_pad=(sr*target_time) - len(signal)
            a = random.randint(0, len_to_pad)
            b = len_to_pad - a
            audio = np.pad(signal, (a,b), "constant")
            under_r_samples.append((audio,code))

    else: # preprocessing actions for overer-represented classes 
        # pad randomly
        len_to_pad=(sr*target_time) - len(signal)
        a = random.randint(0, len_to_pad)
        b = len_to_pad - a
        audio = np.pad(signal, (a,b), "constant")
        over_r_samples.append((audio,code))

  below_record_samples  = under_r_samples + over_r_samples

  return below_record_samples

# COMPILE FINAL TRAIN SET 

def final_set(above_record_samples,below_record_samples):
  '''
  Regroups both preprocessed datasets into the final train, val or test set

  Inputs
  above_record_samples: list of tuples of signal padded issued from ***_split_above_samples functions
  below_record_samples: list of tuples of signal padded issued from ***_split_below_samples functions

  Outputs
  preprocessed train set
  '''
  return above_record_samples+below_record_samples

# PREPROCESSING VALIDATION AND TEST SIGNALS THAT ARE ABOVE TARGET_TIME

def val_test_split_above_samples(above, target_time, sr):
  '''
  Performs preprocessing actions on either the val_set or test containing signals above the target_time

  Inputs
  above      : list of tuples of signal and code above target_time
  target_time: duration in seconds wanted for the signal
  sr         : sampling rate

  Ouputs
  above_record_samples: list of tuples of signal and code of duration target_time
                 
  '''
  above_samples=[]

  # cut into target_time seconds consecutive slices + pad the last one if it is >= target_time - 1
  for signal, code in above:
    nb_split_samples = len(signal) // (target_time*sr) 

    for i in range(0,nb_split_samples):
      sound_i = signal[i*(target_time*sr): (i+1)*(target_time*sr)]
      above_samples.append((sound_i, code))

      if (len(signal) % (target_time*sr)) / sr >= target_time - 1 :
        len_to_pad=(sr*target_time) - len(signal[(nb_split_samples) * (target_time * sr):])
        a = random.randint(0, len_to_pad)
        b = len_to_pad - a
        audio = np.pad(signal[(nb_split_samples)*(target_time*sr):], (a,b), "constant")
        above_samples.append((audio, code))

  return above_samples

# PREPROCESSING VALIDATION AND TEST SIGNALS THAT ARE BELOW TARGET_TIME

def val_test_split_below_samples(below, target_time, sr):
  '''
  Performs preprocessing actions on either the val_set or test containing signals below the target_time

  Inputs
  below      : list of tuples of signal and code below or equal to target_time
  target_time: duration in seconds wanted for the signal
  sr         : sampling rate

  Ouputs
  below_record_samples: list of tuples of signal padded randomly and code 
  '''
  below_record_samples=[]

  # pad the signal randomly
  for signal, code in below:
    len_to_pad=(sr*target_time) - len(signal)
    a = random.randint(0, len_to_pad)
    b = len_to_pad - a
    audio = np.pad(signal, (a,b), "constant")
    below_record_samples.append((audio,code))

  return below_record_samples

