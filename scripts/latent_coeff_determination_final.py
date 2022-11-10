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

import librosa.display

def convert_audio_for_model(user_file, EXPECTED_SAMPLE_RATE):
  audio = AudioSegment.from_file(user_file)
  audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE).set_channels(1)
  output_file = user_file.split('.')[0]+'_converted_audio_file.wav'
  audio.export(output_file, format="wav")
  return output_file

def plot_stft(x_stft_db, sample_rate, file):
    
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

def quantile_data(data, quantile, low_freq, high_freq, center_freq):
    
    data = data.copy(deep=True)
    data = data.iloc[low_freq:high_freq, :]
    data[data < data.loc[center_freq,:].quantile(quantile)] = -80
    
    return data

def extract_signal(data, low_freq, high_freq, x_start, x_end, foward_span=20, reverse_span=10):
    
    data = data.copy(deep=True)
    data = data + abs(data.min().min())  
    data = data.loc[low_freq:high_freq, x_start:x_end]
    
    fig, ax = plt.subplots(2,1, figsize=(6,4))
    sns.heatmap(data, ax=ax[0],  cbar=False)
    ax[0].invert_yaxis()
    
    sig = data.sum(axis=0)
    sig = sig/sig.max()

    sig_smooth = sig.ewm(span=foward_span).mean()[::-1].ewm(span=reverse_span).mean()[::-1]
    sig = sig - sig_smooth.min()
    sig_smooth = sig_smooth - sig_smooth.min()

    ax[1].plot(sig, label='raw', color='grey', alpha=.5)
    ax[1].plot(sig_smooth, label='smoothened', color='darkcyan')
    
    ax[1].set_xlim([data.columns[0], data.columns[-1]])
    plt.subplots_adjust(hspace=.5)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)    

    return pd.DataFrame(sig_smooth)

def add_coeff(data, start, end, coeff_name, coeff_ts, decay=10, pad=64):
    
    data = data.copy(deep=True)
    
    data[coeff_name] = 0
    data[coeff_name].loc[start:(end-1)] = coeff_ts

    data[coeff_name].loc[(start-(decay+pad)):start] = data[coeff_name].loc[(start-(decay+pad)):start][::-1].ewm(span=decay).mean()[::-1]
    data[coeff_name].loc[(end-1):(end+(decay+pad))] = data[coeff_name].loc[(end-1):(end+(decay+pad))].ewm(span=decay).mean()
    
    return data

def get_perc_peaks(perc_df, freq):
    
    perc_df = perc_df.copy(deep=True)
    perc_df = perc_df + abs(perc_df.min())
    
    perc_df.iloc[freq,:].plot()
    
    perc_ts = perc_df.iloc[freq,:]
    perc_ts = perc_ts / (perc_ts.max())
    
    return perc_ts

def set_block_to_zero(data, mid):
    data = data.copy(deep=True)
    data.loc[:, (mid-10):(mid+10)] = -80
    return data

''' 
    The first 3 time-series extracted are persistent, they are active throughout
    the entire animation (high_background, low_background, end_harm)
    -
    -
    -
    
    - JWST A composition of rectangular shapes, vector art, floating glass crystals separated by whitespace, david rudnick, clean, monochromatic ink, artstation, deviantart, pinterest
    - A beautiful collage of a black colorful bliss explosion.  by Ossip Zadkine, by Ian McQue improvisational, neon acrylic colors, digital art, realm of the gods, rave, minimalistic, JWST vector art
    - the realm of gods of infinite bliss, digital art
    
    -
    -
    -
    
    The next 2 are only active in the first third of the animation
    (high_sin_early, low_sin_early)
    -
    -
    
    The following 2 are only active in the second third
    (base_drums, snare_drums)
    - realistic photo of colorful blast, high colored texture, JWST, dark smooth background, very sharp focus, in the style of greg rutswoski, very hyper realistic, highly detailed, fantasy art station
    - A beautiful collage of a black hole consuming a star.  by Ossip Zadkine, by Ian McQue improvisational, digital art, realm of the gods, JWST vector art
    
    The final 3 are used in the final third
    (perc, final_base, final_snare)
    -
    -
    -
    
    step count scaled up in each part 20 - 50
    
    prompts:
        'JWST Futuristic lasers tracing, mondrian, neon fluorescent colors, hannah af klint perfect geometry'
        'JWST Futuristic lasers tracing, colorsmoke, pyramid hoodvisor, raindrops, wet, oiled, kaws, mondrian, hannah af klint perfect geometry abstract acrylic, octane hyperrealism photorealistic airbrush collage painting, monochrome, neon fluorescent colors, minimalist, rule of thirds, retro-scifi cave'
        'JWST AI utopian datacenter or schema, datapipeline or neuron, sentient landscape, omnipotent presence, cybernetic detailed architecture, heavenly aura, painting by Jules Julien and Lisa Frank and Peter Mohrbacher and Alena Aenami and Dave LaChapelle muted colors with minimalism, hauntingly beautiful, vibrant dreamscape, mc escher, trending on artstation, cinematic composition , ultra-detailed'
    
    First 3 time series control different prompts by section?
    
    potentials:
        - 'A psychedelic illusion mountain scenery, moody, space, colorful, JWST sun, artstation, digital art, vector art, david rudnick, futurism'
        
        
Prompt orders

    # PART I
    # (high_background, low_background, high_sin_early, low_sin_early, end_harm)
    
    # base
    'neon fluorescent colors,  by Ian McQue improvisational,  minimalist,  JWST vector art,  rule of thirds,  highly detailed,  rainbow gradient lighting,  heavenly aura',
    
    # high_background
    #'neon fluorescent colors,  pyramid hoodvisor,   by Ian McQue improvisational,  raindrops,  wet,  minimalist,  kaws,   rainbow gradient lighting,  mondrian,',
 
    # low_background
    #'neon fluorescent colors, raindrops,  pyramid hoodvisor, wet,  raindrops, kaws,  wet, hannah af klint perfect geometry abstract acrylic,   rainbow gradient lighting, octane hyperrealism photorealistic airbrush collage painting, JWST vector art of stars, monochrome, art deco, rule of thirds, no fine details, gradients,'
    #'raindrops,   by Ian McQue improvisational, wet,  minimalist, oiled,  kaws, mondrian,  mondrian, octane hyperrealism photorealistic airbrush collage painting, black background with stars, neon fluorescent colors, art deco, retro-scifi cave, noir, clean',
    
    # high_sin_early
    #'pyramid hoodvisor, abstract flames,    by Ian McQue improvisational,  peaceful melody,   minimalist,  harmony aesthetics,   kaws,  firey gradients,    rainbow gradient lighting,  by Ian McQue improvisational,   mondrian,  golden light,  Burning clouds,  forms made of fire, '
    #'neon fluorescent colors, abstract flames,    by Ian McQue improvisational,  heavenly objects,   raindrops,  peaceful melody,   wet,  harmony aesthetics,   kaws,  firey gradients,  Burning clouds,  warm heavenly aura,  thunder and fire rain,  forms made of fire, '
    
    # low_sin_early
    #'neon fluorescent colors, JWST Futuristic lasers tracing,   JWST vector art,  mondrian,   rule of thirds,  octane hyperrealism photorealistic airbrush collage painting,   highly detailed,  retro-scifi cave'
    
    # end_harm
    'JWST Futuristic lasers tracing, colorsmoke, pyramid hoodvisor, raindrops, wet, oiled, kaws, mondrian, hannah af klint perfect geometry abstract acrylic, octane hyperrealism photorealistic airbrush collage painting, monochrome, neon fluorescent colors, minimalist, rule of thirds, retro-scifi cave, clean, gradients',


    # PART II
    # (high_background, low_background, mid_piano, end_harm, base_drums, snare_drums)

    # base
    #'on the galactic shore, JWST vector art of stars, black background with stars, monochrome, minimalistic, art deco, noir, no fine details, no solid shapes',
    
    # high_background
    #'A beautiful collage of a black colorful bliss explosion.  by Ossip Zadkine, by Ian McQue improvisational, neon acrylic colors, digital art, realm of the gods, minimalistic, JWST vector art, depth perspective effect',
    
    # low_background
    #',  JWST vector art,  by Ossip Zadkine,  udnie monochrome,  by Ian McQue improvisational,  minimalistic,  highly detailed,  noir,  ominpotent presence,  no fine details,  heavenly aura,  black negative space,  rainbow gradient lighting'
    
    # mid_piano
    #'a fine colorful mist, minimalist Futuristic lasers tracing, JWST rainbow gradient, colorsmoke, rave',
    
    # end_harm
    #'A beautiful collage of a black neon colorful blast.  by Ossip Zadkine, by Ian McQue improvisational, digital art, realm of the gods, JWST vector art, highly detailed, sentient, ominpotent presence, all knowing, heavenly aura, rainbow gradient lighting, rainbow colors',
    
    # base_drums
    'realistic photo of colorful blast, high colored texture, dark smooth background, very sharp focus, in the style of greg rutswoski, very hyper realistic, highly detailed, fantasy art station',
    
    # snare_drums
    #'on the galactic shore, A beautiful collage of a black neon colorful blast,  by Ossip Zadkine,  by Ossip Zadkine,  primarily black compisition,  realm of the gods,  noir,  all knowing,  no solid shapes,  heavenly aura,  black negative space,  rainbow gradient lighting',

    # Part III
    # (high_background, low_background, end_harm, perc, final_base, final_snare)
    
    # high_background
    #'JWST Futuristic lasers tracing,  by Ossip Zadkine,  pyramid hoodvisor,  by Ian McQue improvisational,  raindrops,  digital art,  kaws,  realm of the gods,  mondrian,  JWST vector art,  hannah af klint perfect geometry abstract acrylic,  highly detailed,  by Ian McQue improvisational,  ominpotent presence,  octane hyperrealism photorealistic airbrush collage painting,  heavenly aura,  monochrome,  labyrinth,  minimalist,  mc escher,  rule of thirds,  3D hyperbolic,  retro-scifi cave,  intricate',
    
    # low_background
    'A beautiful collage of a black neon colorful blast.  by Ossip Zadkine, by Ian McQue improvisational, digital art, realm of the gods, JWST vector art, highly detailed, sentient, ominpotent presence, all knowing, heavenly aura, rainbow gradient lighting',
    
    # end_harm
    # perc
    
    # high_sin_late
    ', on the galactic shore, A beautiful collage of a black neon colorful blast,  JWST vector art,  by Ossip Zadkine,  by Ossip Zadkine,  by Ian McQue improvisational,  by pete mondrian,  highly detailed,  minimalistic,  all knowing,  no fine details,  heavenly aura'
    
    # final_base
    # final_snare

    # neon fluorescent colors, rainbow gradient lighting
    
    - A beautiful collage of a black neon colorful blast, by Ossip Zadkine, by Ian McQue improvisational, digital art, realm of the gods, 
    
'''

#extract_signal(other_df, 24, 25, 15, 3500, 5, 15)

#%%
''' To get background  '''
EXPECTED_SAMPLE_RATE = 16382
project_path = '/Users/williamstanford/Desktop/Fall2022/Neural_Rendering/'

# Other.mp3 was generated from using a demucs google colab to separate out 
# different components of the audio, other refers to non-voice + non-bass + non-drums
other_file = project_path + '/demucs_separated/mdx_extra/RichKid_DivKid/other.mp3' 
x_stft_db_other, other_df = convert_and_plot_spec(other_file, EXPECTED_SAMPLE_RATE)

ts_df = pd.DataFrame(index=other_df.columns)

high_data = quantile_data(other_df, .5, 38, 40, 39)
high_background = extract_signal(high_data, 38, 39, 0, other_df.shape[1], 20, 20)
ts_df = add_coeff(ts_df, 0, other_df.shape[1], 'high_background', high_background[0])

low_data = quantile_data(other_df, .5, 23, 27, 25)
low_background = extract_signal(low_data, 23, 27, 0, other_df.shape[1], 20, 20)
ts_df = add_coeff(ts_df, 0, other_df.shape[1], 'low_background', low_background[0])


#%%
''' harmony that ends each section '''

end_data = quantile_data(other_df, .8, 62, 90, 77)
end_harm = extract_signal(end_data, 62, 90, 0, other_df.shape[1], 20, 20)
ts_df = add_coeff(ts_df, 0, other_df.shape[1], 'end_harm', end_harm[0])

#%%
''' For oscillating harmony '''

EXPECTED_SAMPLE_RATE = 16382
project_path = '/Users/williamstanford/Desktop/Fall2022/Neural_Rendering/'
file = project_path + 'audio/RichDivKid.mp3'

x_stft_db, spec_df = convert_and_plot_spec(file, EXPECTED_SAMPLE_RATE)

high_sin_early = np.sin(2*np.pi*((np.arange(390, 1442)-390)/370) + 0.70*2*np.pi)
ts_df = add_coeff(ts_df, 390, 1442, 'high_sin_early', high_sin_early)


low_sin_early = np.sin(2*np.pi*((np.arange(730, 1442)-730)/370) + 0.70*2*np.pi)
ts_df = add_coeff(ts_df, 730, 1442, 'low_sin_early', low_sin_early)


#%% 
''' To extract electric piano starting around 1:07 '''

EXPECTED_SAMPLE_RATE = 16382
project_path = '/Users/williamstanford/Desktop/Fall2022/Neural_Rendering/'
other_file = project_path + '/demucs_separated/mdx_extra/RichKid_DivKid/other.mp3'
x_stft_db_other, other_df = convert_and_plot_spec(other_file, EXPECTED_SAMPLE_RATE)

piano_mid_df = other_df.copy(deep=True)
piano_mid_df.loc[100:1200, 2115:2164] = -80
piano_mid_df.loc[110:1200, 2174:2185] = -80
piano_mid_df.loc[120:1200, 2195:2257] = -80
piano_mid_df.loc[130:1200, 2272:2290] = -80
piano_mid_df.loc[140:1200, 2305:2348] = -80
piano_mid_df.loc[150:1200, 2370:2381] = -80
piano_mid_df.loc[160:1200, 2404:2440] = -80
piano_mid_df.loc[170:1200, 2460:2470] = -80
piano_mid_df.loc[180:1200, 2460:2470] = -80
piano_mid_df.loc[190:1200, 2498:2528] = -80
piano_mid_df.loc[200:1200, 2551:2561] = -80
piano_mid_df.loc[210:1200, 2590:2620] = -80
piano_mid_df.loc[220:1200, 2695:2710] = -80
piano_mid_df.loc[230:1200, 2767:2798] = -80
piano_mid_df.loc[100:1200, 2860:2900] = -80

mid_piano = extract_signal(piano_mid_df, 100, 1200, 2140, 2900, 5, 15)
ts_df = add_coeff(ts_df, 2140, 2900, 'mid_piano', mid_piano[0])

#%%
''' To extract different mid-drum parts '''

EXPECTED_SAMPLE_RATE = 16382
project_path = '/Users/williamstanford/Desktop/Fall2022/Neural_Rendering/'
drum_file = project_path + '/demucs_separated/mdx_extra/RichKid_DivKid/drums.mp3'
x_stft_db_drums, drum_df = convert_and_plot_spec(drum_file, EXPECTED_SAMPLE_RATE)

base_data = quantile_data(drum_df, .7, 0, 12, 7)

base_drums = extract_signal(base_data, 2, 11, 1430, 3000, 2, 5)
ts_df = add_coeff(ts_df, 1430, 3000, 'base_drums', base_drums[0])


snare_df = drum_df.copy(deep=True)
snare_df = snare_df.iloc[:30, 1435:3100]

blocks = [1447, 1500, 1535, 1543, 1593, 1627, 1636, 1675, 1720, 1726, 1761,
          1865, 1904, 1956, 1990, 1999, 2040, 2086, 2226, 2265, 2317, 2352, 
          2360, 2400, 2440, 2450, 2486, 2540, 2585, 2625, 2675, 2710, 2720,
          2760, 2805, 2810, 2840, 2845]

for block in blocks:
    snare_df = set_block_to_zero(snare_df, block)

snare_drums = extract_signal(snare_df, 14, 30, 1437, 3000, 2, 5)
ts_df = add_coeff(ts_df, 1430, 3000, 'snare_drums', snare_drums[0])


#%%
''' To separate out percussive components '''

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

perc_data = extract_signal(perc_df, 50, 500, 3030, 4510, 3, 3)
ts_df = add_coeff(ts_df, 3030, 4510, 'perc', perc_data[0])


#%%
''' To extract different ending drum parts '''

base_end = quantile_data(drum_df, .8, 0, 12, 7)

base_end = extract_signal(base_end, 2, 11, 4500, 5350, 2, 5)
ts_df = add_coeff(ts_df, 4500, 5350, 'final_base', base_end[0])


snare_end_df = drum_df.copy(deep=True)
snare_end_df = snare_end_df.iloc[:30, 4500:5350]

blocks = [4520, 4570, 4614, 4663, 4700, 4708, 4750, 4790, 4798, 4825, 4835,
          4920, 4932, 4974, 5025, 5065, 5068, 5110, 5150, 5155, 5190, 5195
    ]

for block in blocks:
    snare_end_df = set_block_to_zero(snare_end_df, block)


snare_end = extract_signal(snare_end_df, 14, 30, 4500, 5350, 2, 5)
ts_df = add_coeff(ts_df, 4500, 5350, 'final_snare', snare_end[0])

#%%
def plot_spec_feat(data):
    data = data.copy(deep=True)
    data = data + abs(data.min())
    data.fillna(0, inplace=True)
    
    fig, ax = plt.subplots(1,1, figsize=(6,4))
    sns.heatmap(data, ax=ax, cbar=False )
    ax.invert_yaxis()
    
#%%
plot_spec_feat(harm_df.iloc[:100,300:1200])

#%%
import random

def prompt_mixer(prompt1, prompt2, fraction):
    
    tokens1 = prompt1.split(',')
    tokens2 = prompt2.split(',')
    
    
    num_tokens1 = int(np.floor(fraction*len(tokens1)))
    num_tokens2 = int(np.floor(fraction*len(tokens2)))
    
    num_tokens = min([num_tokens1, num_tokens2])
    
    rand_tokens1 = random.sample(tokens1, num_tokens)
    rand_tokens2 = random.sample(tokens2, num_tokens)
    
    rand_tokens1 = [
    tokens1[i] for i in sorted(random.sample(range(len(tokens1)), num_tokens))
    ]

    rand_tokens2 = [
    tokens2[i] for i in sorted(random.sample(range(len(tokens2)), num_tokens))
    ]
    
    mixed_prompt = "'"
    
    for tok1, tok2 in zip(rand_tokens1, rand_tokens2):
        
        rand_i = random.randint(0, 1)
        
        if rand_i == 0:
            mixed_prompt = mixed_prompt + tok1 + ", " + tok2 + ", "
        else:
            mixed_prompt = mixed_prompt + tok1 + ", " + tok2 + ", "
    
    mixed_prompt = mixed_prompt + "',\n"
    
    print(mixed_prompt)
    
    
#prompt1 = 'JWST Futuristic lasers tracing, colorsmoke, pyramid hoodvisor, raindrops, kaws, mondrian, hannah af klint perfect geometry abstract acrylic, by Ian McQue improvisational, octane hyperrealism photorealistic airbrush collage painting, monochrome, neon fluorescent colors, minimalist, rule of thirds, retro-scifi cave, rainbow gradient lighting'
prompt1 = 'neon fluorescent colors,  pyramid hoodvisor,   by Ian McQue improvisational,  raindrops,  wet,  minimalist,  kaws,   rainbow gradient lighting,  mondrian, Burning clouds, thunder and fire rain,'
prompt2 = 'abstract flames, heavenly objects, peaceful melody, harmony aesthetics, firey gradients, by Ian McQue improvisational, warm heavenly aura, golden light, forms made of fire, forms of gold'

for i in range(20):
    prompt_mixer(prompt1, prompt2, .75)
    
#%%
p = 'JWST Futuristic lasers tracing, colorsmoke, pyramid hoodvisor, raindrops, wet, oiled, kaws, mondrian composition 3d depth, hannah af klint perfect geometry abstract acrylic, octane hyperrealism photorealistic airbrush collage painting, monochrome, neon fluorescent colors, minimalist, rule of thirds, retro-scifi cave, clean, rainbow gradients, monolithic monument, heavenly aura, omniopotent presence'

print(len(p.split(', ')))
    
#%%
'pyramid hoodvisor, A beautiful collage of a black neon colorful blast,  oiled,  by Ossip Zadkine,  kaws,  by Ian McQue improvisational,  mondrian,  digital art,  octane hyperrealism photorealistic airbrush collage painting,  JWST vector art,  minimalist,  highly detailed,  rule of thirds,  ominpotent presence,  retro-scifi cave,  all knowing,  rainbow gradient lighting,  heavenly aura'

' A beautiful collage of a black neon colorful blast,  pyramid hoodvisor,  digital art,  wet,  realm of the gods,  kaws,  JWST vector art,  mondrian,  highly detailed,  monochrome,  ominpotent presence,  minimalist,  all knowing,  rule of thirds,  heavenly aura,  rainbow gradient lighting,  rainbow gradient lighting'
3911945254

#%%
EXPECTED_SAMPLE_RATE = 16382
audio_files = glob.glob('/Users/williamstanford/Desktop/Fall2022/Neural_Rendering/demucs_separated/mdx_extra/RichKid_DivKid/*')
audio_files.sort()

for file in audio_files:
    if '_converted_audio_file.wav' not in file:
        _, _ = convert_and_plot_spec(file, EXPECTED_SAMPLE_RATE)
        print(file.split('/')[-1])


