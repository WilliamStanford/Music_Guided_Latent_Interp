import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import display as librosadisplay

from pydub import AudioSegment
from scipy.io import wavfile
import seaborn as sns

import glob
import os 


def convert_audio_for_model(user_file, EXPECTED_SAMPLE_RATE):
  audio = AudioSegment.from_file(user_file)
  audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE).set_channels(1)
  output_file = user_file.split('.')[0]+'_converted_audio_file.wav'
  audio.export(output_file, format="wav")
  return output_file

def plot_stft(x_stft_db, sample_rate):
    
  fig, ax = plt.subplots()
  fig.set_size_inches(6, 3)
  librosadisplay.specshow(data=x_stft_db, y_axis='log', sr=sample_rate, cmap='gray_r')
  ax.set_xlabel(file.split('/')[-1])


def convert_and_plot_spec(file, EXPECTED_SAMPLE_RATE):

    
    if not os.path.exists(file.split('.')[0]+'_converted_audio_file.wav'):
    
        y, sr = librosa.load(file, sr=EXPECTED_SAMPLE_RATE)
        
        # Converting to the expected format for the model
        # in all the input 4 input method before, the uploaded file name is at
        # the variable uploaded_file_name
        converted_audio_file = convert_audio_for_model(file, EXPECTED_SAMPLE_RATE)
        
    else:
        converted_audio_file = file.split('.')[0]+'_converted_audio_file.wav'
        
    # Loading audio samples from the wav file:
    sample_rate, audio_samples = wavfile.read(converted_audio_file, 'rb')
    
    MAX_ABS_INT16 = 32768.0
    
    x_stft = np.abs(librosa.stft(audio_samples / MAX_ABS_INT16, n_fft=2048))
    x_stft_db = librosa.amplitude_to_db(x_stft, ref=np.max)
    spec_df = pd.DataFrame(x_stft_db)
    
    plot_stft(x_stft_db, EXPECTED_SAMPLE_RATE, file)
    
    return x_stft_db, spec_df

def add_coeff(data, start, end, coeff_name, coeff_ts, decay=10, pad=64):
    
    data = data.copy(deep=True)
    
    data[coeff_name] = 0
    data[coeff_name].loc[start:(end-1)] = coeff_ts

    data[coeff_name].loc[(start-(decay+pad)):start] = data[coeff_name].loc[(start-(decay+pad)):start][::-1].ewm(span=decay).mean()[::-1]
    data[coeff_name].loc[(end-1):(end+(decay+pad))] = data[coeff_name].loc[(end-1):(end+(decay+pad))].ewm(span=decay).mean()
    
    return data


#%%
EXPECTED_SAMPLE_RATE = 16382
project_path = '/Users/williamstanford/Desktop/Fall2022/Neural_Rendering/'
#file = project_path + 'audio/Dreamer_DivKid.mp3'
#file = project_path + 'audio/EmptyMoons_DivKid.mp3'
file = project_path + 'audio/RichDivKid.mp3'

x_stft_db, spec_df = convert_and_plot_spec(file, EXPECTED_SAMPLE_RATE)

ts_df = pd.DataFrame(index=spec_df.columns)


#%%

high_sin_early = np.sin(2*np.pi*((np.arange(390, 1442)-390)/370) + 0.70*2*np.pi)
ts_df = add_coeff(ts_df, 390, 1460, 'high_sin_early', high_sin_early)


low_sin_early = np.sin(2*np.pi*((np.arange(730, 1442)-730)/370) + 0.70*2*np.pi)
ts_df = add_coeff(ts_df, 730, 1460, 'low_sin_early', low_sin_early)

#%%

def extract_signal(data, low_freq, high_freq, x_start, x_end):
    
    data = data.copy(deep=True)
    data = data + abs(data.min().min())
    
    data = data.loc[low_freq:high_freq, x_start:x_end]
    
    sig = data.sum(axis=0)
    
    sig = sig.ewm(span=20).mean()[::-1].ewm(span=10).mean()[::-1]
    sig = sig - sig.min()
    sig = sig/sig.max()
    
    fig, ax = plt.subplots(2,1, figsize=(6,3))
    sns.heatmap(data, ax=ax[0])
    
    ax[1].plot(sig)
    
    plt.subplots_adjust(hspace=.75)
    
    
    return pd.DataFrame(sig)

ts_df['early_keys'] = 0

freqs = {
    73:{'start':1080, 'end':1188, 'coeff':.08},
    65:{'start':1150, 'end':1320, 'coeff':.12},
    78:{'start':1250, 'end':1350, 'coeff':.2},
    87:{'start':1330, 'end':1442, 'coeff':.23},
    }

for freq in freqs.keys():

    width=2
    sig =  extract_signal(spec_df, freq-width, freq+width, freqs[freq]['start'], freqs[freq]['end'])
    
    for ind in sig.index:
        ts_df.loc[ts_df.index==ind, 'early_keys'] = freqs[freq]['coeff']*sig.loc[ind, 0]

ts_df['early_keys'] = ts_df['early_keys'].ewm(span=10).mean()

fig, ax = plt.subplots(1,1, figsize=(6,3))
ts_df['early_keys'].iloc[900:1500].plot(ax=ax)
    

#%%
import numpy as np
import librosa.display

def get_perc_peaks(perc_df, freq):
    
    perc_df = perc_df.copy(deep=True)
    perc_df = perc_df + abs(perc_df.min())
    
    perc_df.iloc[freq,:].plot()
    
    perc_ts = perc_df.iloc[freq,:]
    perc_ts = perc_ts / (perc_ts.max())
    
    return perc_ts

project_path = '/Users/williamstanford/Desktop/Fall2022/Neural_Rendering/'
file = project_path + 'audio/RichDivKid.mp3'
y, sr = librosa.load(file, sr=16382)

D = librosa.stft(y)

D_harmonic, D_percussive = librosa.decompose.hpss(D)

harmonic = librosa.amplitude_to_db(np.abs(D_harmonic))

harm_df = pd.DataFrame(librosa.amplitude_to_db(np.abs(D_harmonic)))
perc_df = pd.DataFrame(librosa.amplitude_to_db(np.abs(D_percussive)))

harm_df.to_csv(project_path+file.split('.')[0].split('/')[-1]+'_harmonic.csv')
perc_df.to_csv(project_path+file.split('.')[0].split('/')[-1]+'_percussion.csv')

ts_df['perc'] = get_perc_peaks(perc_df, 30)
ts_df['perc'] = ts_df['perc'].ewm(span=5).mean()

#%%


def plot_spec_feat(data):
    data = data.copy(deep=True)
    data = data + abs(data.min())
    data.fillna(0, inplace=True)
    
    fig, ax = plt.subplots(1,1, figsize=(6,3))
    sns.heatmap(data, ax=ax)
    ax.invert_yaxis()
    

plot_spec_feat(harm_df.iloc[60:100,:1400])


#%%
EXPECTED_SAMPLE_RATE = 16382
project_path = '/Users/williamstanford/Desktop/Fall2022/Neural_Rendering/'
#file = project_path + 'audio/Dreamer_DivKid.mp3'
#file = project_path + 'audio/EmptyMoons_DivKid.mp3'
drum_file = project_path + '/demucs_separated/mdx_extra/RichKid_DivKid/drums.mp3'

x_stft_db_drums, drum_df = convert_and_plot_spec(drum_file, EXPECTED_SAMPLE_RATE)


#%%

plot_spec_feat(drum_df.iloc[:30,1600:1800])

#%%
def partial_sin(x, rad1, rad2):
    
    y = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        
        xi_mod = xi % 2*np.pi
        
        if xi_mod > rad1 and xi_mod < rad2:
            y[i] = xi
            
        else:
            y[i] = 0
            
    return y

x = np.arange(1500,  3000)
high_sin_partial = (x-390)/370

plt.plot(x, np.sin(2*np.pi*partial_sin(high_sin_partial, np.pi/4, 3*np.pi/4)))
            
            
    


#%%
EXPECTED_SAMPLE_RATE = 16382
audio_files = glob.glob('/Users/williamstanford/Desktop/Fall2022/Neural_Rendering/demucs_separated/mdx_extra/RichKid_DivKid/*')
audio_files.sort()

for file in audio_files:
    if '_converted_audio_file.wav' not in file:
        _, _ = convert_and_plot_spec(file, EXPECTED_SAMPLE_RATE)
        print(file.split('/')[-1])

#%%
fig, ax = plt.subplots(1,1, figsize=(6,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.plot(np.arange(0, end), high_sin, color='darkcyan')
ax.text(0, .05, 'exponential drop', fontsize=10, color='black')
ax.set_ylabel('Coefficient')
ax.set_xlabel('Frame (32 fps, 45 seconds total)')
fig.savefig(project_path + 'figures/latent_coeff_high_sin.png', bbox_inches='tight', dpi=300)
 
ts_df['high_sin'] = high_sin

#%%
low_sin = np.zeros(end)
low_sin[s_63:1440] = -0.02
ls = pd.DataFrame(low_sin)
ls['ewm'] = ls[0][::-1].ewm(span=10).mean()
ls['r_ewm'] = ls['ewm'][::-1]
low_sin[:s_63] = ls['r_ewm'].iloc[:s_63]
low_sin[s_63:1440] = 0.025*np.sin(2*np.pi*((x_sin_63-s_63)/370) + 0.70*2*np.pi)
low_sin = low_sin*7
low_sin[1439:1462] = np.arange(-0.115, 0, 0.005)

fig, ax = plt.subplots(1,1, figsize=(6,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.plot(np.arange(0, end), low_sin, color='mediumspringgreen')
ax.text(270, .025, 'exponential drop', fontsize=10, color='black')
ax.set_ylabel('Coefficient')
ax.set_xlabel('Frame (32 fps, 45 seconds total)')
fig.savefig(project_path + 'figures/latent_coeff_low_sin.png', bbox_inches='tight', dpi=300)

ts_df['low_sin'] = low_sin


#%%

