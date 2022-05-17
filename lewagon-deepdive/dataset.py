import pandas as pd
import os
import librosa
import random
import numpy as np

def get_dataset(path_to_watkins_sound_list, families_to_keep=[], species_to_keep=[], multi_species=False, noise = False, min_duration = 0):
    """
    Load the watkins_sound_list and return a df with the sounds we want to have in the dataset.

    Input:
    path_to_watkins_sound_list : path to watkins_sound_list.csv file.
    families_to_keep : list of families that we want to include in the dataset. Default: [] means all families.
    species_to_keep : list of species that we want to include in the dataset. Default: [] means all species.
    multi_species : True if you want to include sounds with several species. Default: False.
    noise : True if you want to include sounds with polluting noise. Default: False.
    min_duration : minimum duration (seconds) of the sounds you want to include in the dataset. Default: 0.

    Output:
    Creates a csv file with all the selected sounds
    Returns a dataframe with the sounds we want to keep.
    """

    # strings for the name of the final csv file to be created
    family_string = 'all'
    species_string = 'all'
    nb_species = 'multi'
    included_noise = 'polluting'

    # loading the watkins_sound_list in a dataframe
    dataset = pd.read_csv(path_to_watkins_sound_list)

    # keep only required families
    if families_to_keep:
        dataset = dataset[dataset.family_code.isin(families_to_keep)]
        family_string = '-'.join(families_to_keep)

    #keep only required species
    if species_to_keep:
        dataset = dataset[dataset.species_code.isin(species_to_keep)]
        species_string = '-'.join(species_to_keep)

    # if multi_species is False, keep only sounds with one species
    if not multi_species:
        dataset = dataset[dataset.multi_species==False]
        nb_species = 'mono'

    # if noise is False, keep only sounds with no polluting noise
    if not noise:
        dataset = dataset[dataset.noise==False]
        included_noise='no'

    # keep only sounds longer that min_duration
    if min_duration != 0 :
        dataset = dataset[dataset.duration >= min_duration]

    # reseting the index
    dataset.reset_index(drop=True,inplace=True)

    # create csv file
    csv_name = f"families-{family_string}_species-{species_string}_{nb_species}-species_{included_noise}-noise_min-{min_duration}sec"
    dataset.to_csv(f'/content/drive/MyDrive/lewagon-deepdive/raw_data/{csv_name}.csv')

    return dataset

def get_audio_data_and_species_code(audio_directory, filename, species_code, target='family', sr=44_100):
    '''
    Load an audio file and return a tuple with the audio data and species code

    Input:
    audio_directory : path to the directory where audio is stored
    filename : name of the audio file
    species_code : code of the species recorded on the audio file
    sr : sampling rate. Default = 44100 Hz

    Output:
    audio_data : np.array of the audio time series. Multi-channel is supported. shape=(n,) or (…, n). See librosa.load() documentation.
    species_code : the species code
    '''

    # path to audio file (audio_directory and filename)
    filepath = os.path.join(audio_directory,filename)

    # load the audio file
    audio_data, sampling_rate = librosa.load(filepath,sr=sr)

    if target == 'family':
            species_code=species_code[:2]

    return audio_data, species_code


def get_list_of_tuples(dataset, audio_directory, target='family', sr=44_100, nb_rows=None):
    '''
    Takes a pandas dataframe containing the names of the sound files to be treated.
    Returns a list of tuples of two items:
    - the data of the audio file
    - the corresponding species code

    Input:
    dataset : dataframe created by function get_dataset()
    audio_directory : path to the directory where audio is stored
    sr : sampling rate (default = 44100 Hz)
    nb_rows : number of rows to iterate over in the csv (to allow testing on small number of rows)

    Output:
    list_of_tuples :  a list of tuples containing the data and the species code
    '''

    # resize the dataset if requested in parameters
    if nb_rows:
        dataset = dataset.head(nb_rows)

    # iterate over the rows of the dataset to get the audio data and the species code for each audio file
    list_of_tuples = dataset.apply(lambda row: get_audio_data_and_species_code(audio_directory, row.filename, row.species_code, target, sr),axis=1).tolist()

    return list_of_tuples


def get_mel_spec_array(y,sr):
    ''' 
    Parameters :
    y : audio data
    sr : sampling rate

    Returns :
    S_DB : np.array of the mel spectrogram (log y scale)
    '''

    S = librosa.feature.melspectrogram(y=y, sr=sr) # Compute a mel-scaled spectrogram
    S_DB = librosa.power_to_db(S, ref=np.max) # Convert a power spectrogram (amplitude squared) to decibel (dB) units

    return S_DB

def get_arrays_from_audio(input_list, spectro_type='mel', sr = 44_100):
    '''
    Takes a list of tuples (audio_data,species_code).
    Returns a list of tuples (array,species_code)

    Parameters:
        input_list : list of tuples (audio_data,species)
        sr = sample rate. Default: 44100 Hz
        target: 'family' or 'species'

    Returns:
        list_of_tuples :  list of tuples (array,species)
    '''
    list_of_tuples = []
    for input_tuple in input_list:
        my_list = []
        # compute the correct spectrogram
        if spectro_type == 'mel':
            array = get_mel_spec_array(input_tuple[0],sr)
        elif spectro_type == 'log':
            array = get_log_spec_array(input_tuple[0],sr)
        elif spectro_type == 'lin':
            array = get_lin_spec_array(input_tuple[0],sr)
        else:
            array=None

        # append the array
        my_list.append(array)

        # append target code
        my_list.append(input_tuple[1])

        # convert the list in a tuple
        my_tuple = tuple(my_list)

        # append the tuple to the output list
        list_of_tuples.append(my_list)

    return list_of_tuples