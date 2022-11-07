import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import display as librosadisplay

from IPython.display import Audio, Javascript
from pydub import AudioSegment
from scipy.io import wavfile
import seaborn as sns
from scipy import signal


project_path = '/Users/williamstanford/Desktop/Fall2022/Neural_Rendering/'
#file = project_path + 'audio/Dreamer_DivKid.mp3'
#file = project_path + 'audio/EmptyMoons_DivKid.mp3'
file = project_path + 'audio/RichDivKid.mp3'
y, sr = librosa.load(file, sr=16382)

#%%

EXPECTED_SAMPLE_RATE = 16382

def convert_audio_for_model(user_file, output_file=file.split('.')[0]+'_converted_audio_file.wav'):
  audio = AudioSegment.from_file(user_file)
  audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE).set_channels(1)
  audio.export(output_file, format="wav")
  return output_file

# Converting to the expected format for the model
# in all the input 4 input method before, the uploaded file name is at
# the variable uploaded_file_name
converted_audio_file = convert_audio_for_model(file)

# Loading audio samples from the wav file:
sample_rate, audio_samples = wavfile.read(converted_audio_file, 'rb')

# Show some basic information about the audio.
duration = len(audio_samples)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(audio_samples)}')

#%%
# Let's listen to the wav file.
Audio(audio_samples, rate=sample_rate)
def plot_stft(x_stft_db, sample_rate, xs, ys, colors, version, s_xs, s_alphas, s_y):
    
  fig, ax = plt.subplots()
  fig.set_size_inches(20, 10)
  
  librosadisplay.specshow(data=x_stft_db, y_axis='log', sr=sample_rate, cmap='gray_r')

  if len(xs) > 0:
      for x, y, c in zip(xs, ys, colors):
          ax.plot(x, y, color=c, linewidth=2)
          
  if len(s_xs) > 0:
      for s_x, s_alph in zip(s_xs, s_alphas):
          for xp, a in zip(s_x, s_alph):
              ax.scatter(x, s_y, color='white', linewidth=2, alpha=a)

  plt.colorbar(format='%+2.0f dB')
  

#%%
MAX_ABS_INT16 = 32768.0

x_stft = np.abs(librosa.stft(audio_samples / MAX_ABS_INT16, n_fft=2048))
x_stft_db = librosa.amplitude_to_db(x_stft, ref=np.max)
spec_df = pd.DataFrame(x_stft_db)

ts_df = pd.DataFrame(np.ones(1472), columns=['starting'])
#%%

fig, ax = plt.subplots()
fig.set_size_inches(6, 3)

librosadisplay.specshow(data=x_stft_db, y_axis='log', sr=sample_rate, x_axis='time')
fig.savefig(project_path + 'figures/latent_spectrogram_base_'+file.split('.')[0].split('/')[1]+'.png', bbox_inches='tight', dpi=300)

#%%
plot_stft(x_stft_db, EXPECTED_SAMPLE_RATE, [], [], [], 'melody', [], [], None)
#%%

#%%
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import librosa
import librosa.display

project_path = '/Users/williamstanford/Desktop/Fall2022/Neural_Rendering/'
#file = project_path + 'audio/Dreamer_DivKid.mp3'
file = project_path + 'audio/RichDivKid.mp3'
#file = project_path + 'audio/EmptyMoons_DivKid.mp3'
y, sr = librosa.load(file, sr=16382)

D = librosa.stft(y)

D_harmonic, D_percussive = librosa.decompose.hpss(D)

harmonic = librosa.amplitude_to_db(np.abs(D_harmonic))

harm_df = pd.DataFrame(librosa.amplitude_to_db(np.abs(D_harmonic)))
perc_df = pd.DataFrame(librosa.amplitude_to_db(np.abs(D_percussive)))

harm_df.to_csv(project_path+file.split('.')[0].split('/')[-1]+'_harmonic.csv')
perc_df.to_csv(project_path+file.split('.')[0].split('/')[-1]+'_percussion.csv')


#%%
def plot_spec(data):
    data = data.copy(deep=True)
    data = data + abs(data.min())
    data.fillna(0, inplace=True)
    
    fig, ax = plt.subplots(1,1, figsize=(6,3))
    sns.heatmap(data, ax=ax)
    ax.invert_yaxis()
    

plot_spec(perc_df.iloc[:100,:3090])

plot_spec(harm_df.iloc[:100,:3090])

#%%
perc_df = perc_df + abs(perc_df.min().min())
harm_df.iloc[:,2800:3060].sum(axis=0).plot()

#%%



#%%
def sin_centered(key_freq, width, x, y):
    distance = (width + 1 - np.abs(key_freq - x))*np.abs(np.sin(2*np.pi*y/200))
    return np.max([distance,0])

def linear_decay(key_freq, width, x, y):   
     distance = width + 1 - np.abs(key_freq - x)
     return np.max([distance,0])
 
def cliff_decay(key_freq, width, x, y):
    
    if x > key_freq:
        distance = 0
    else:       
        distance = width - (key_freq - x)
    
    return np.max([distance,0])


def get_mesh_grid(df_audio, func, key_freq, width):
    
    df_audio = df_audio.copy(deep=True)
    df_mesh = pd.DataFrame(np.zeros((df_audio.shape[0], df_audio.shape[1])),
                                     index=list(df_audio.index), 
                                     columns=list(df_audio.columns))
    
    df_audio = df_audio + np.abs(df_audio.min().min())
    
    for x in df_mesh.index:
        for y in df_mesh.columns:

            if abs(x-key_freq) <= width:
                df_mesh.loc[df_mesh.index==x, y] = func(key_freq, width, x, y)
    
    df_mesh = df_mesh/df_mesh.max().max()
    
    prompt_coeffs = pd.DataFrame(df_audio.values*df_mesh.values, columns=df_audio.columns, index=df_audio.index)
    
    return prompt_coeffs

def get_perc_peaks(perc_df, freq):
    
    perc_df = perc_df.copy(deep=True)
    perc_df = perc_df + abs(perc_df.min())
    
    perc_df.iloc[freq,:].plot()
    
    perc_ts = perc_df.iloc[freq,:]
    perc_ts = perc_ts / (perc_ts.max()*7)
    
    return perc_ts
    
links = [30,37]
    
key_freqs = {
    11:{'width':7, 'function':cliff_decay,'smooth':True, 'weighting': 'add'},
    13:{'width':2, 'function':linear_decay,'smooth':True},
    #22:{'width':8, 'function':cliff_decay,'smooth':True, },
    #30:{'width':2, 'function':cliff_decay,'smooth':True},
    37:{'width':10, 'function':cliff_decay,'smooth':True},
    #44:{'width':2, 'function':linear_decay,'smooth':True},
    54:{'width':2, 'function':linear_decay,'smooth':True},
    66:{'width':4, 'function':linear_decay,'smooth':True},
    73:{'width':2, 'function':linear_decay,'smooth':True},
    78:{'width':2, 'function':linear_decay,'smooth':True},
    #90:{'width':4, 'function':linear_decay,'smooth':True},
    #110:{'width':4, 'function':linear_decay,'smooth':True},
    138:{'width':12, 'function':cliff_decay,'smooth':True},
    150:{'width':8, 'function':cliff_decay,'smooth':True},
    }

tx1, tx2, ty1, ty2 = 0, 200, 0, 3060 # harm_df.shape[0], harm_df.shape[1]
prompt_coeffs = pd.DataFrame(np.zeros((int(tx2-tx1), int(ty2-ty1))), 
                              index = np.arange(tx1, tx2),
                              columns = np.arange(ty1, ty2))

#%%
for key in key_freqs:
    
    if 'coeffs' not in list(key_freqs[key].keys()):
        
        print('calculating for key :'+str(key))
        
        func = key_freqs[key]['function']
        width = key_freqs[key]['width']     
        mesh = get_mesh_grid(harm_df.iloc[tx1:tx2,ty1:ty2], func, key, width)
        
        key_freqs[key]['mesh'] = mesh 
        
        ts_mesh = mesh.sum(axis=0)
        ts_mesh = ts_mesh/ts_mesh.max()
        
        if key_freqs[key]['smooth']:
            key_freqs[key]['coeffs'] = ts_mesh.ewm(span=20).mean()[::-1].ewm(span=5).mean()[::-1]
        else:
            key_freqs[key]['coeffs'] = ts_mesh.sum(axis=0)
    
    prompt_coeffs = prompt_coeffs + key_freqs[key]['mesh']
    
#%%
fig, ax = plt.subplots(1,2, figsize=(14,4))
sns.heatmap(harm_df.iloc[0:150,0:1440], ax=ax[0])
sns.heatmap(prompt_coeffs.iloc[0:150,0:1440],cmap='viridis', ax=ax[1])

ax[0].invert_yaxis()
ax[1].invert_yaxis()

plt.subplots_adjust(wspace=0.1, hspace=None)

# 746201461

#%%

def taper_at_ind(ts_df, target_ind, col, thresh=0.05):
    ts_df = ts_df.copy(deep=True)
    
    cont = True
    
    for ind in ts_df.index:  
        if cont == True:     
            if ind > target_ind:
                if ts_df[col].loc[ind:].max() > .15:
                    
                    ts_df[col].loc[ind:] = ts_df[col].loc[ind:]*.99
                    
                else:
                    cont = False
            
    return ts_df

def taper_end(ts_df, target_ind, col, thresh=0.05, factor=.95):
    
    ts_df = ts_df.copy(deep=True)

    for ind in ts_df.index:        
        if ind > target_ind:
            
            ts_df.loc[ts_df.index==ind, col]= ts_df.loc[(int(ind-1)), col] * factor
            
    return ts_df

ts_df = pd.DataFrame(index=ts_mesh.index)

for key in key_freqs.keys():
    
    ts_df[key] = key_freqs[key]['coeffs']
        

used_freqs = [37, 150, 78, 73, 138, 11, 13, 54, 66]

ts_df = ts_df[used_freqs]
ts_df = ts_df*.85

slow = np.zeros(ts_df.shape[0])
slow[-150:] = np.arange(0,.5, .5/150)
ts_df['slow'] = slow
ts_df['perc'] = get_perc_peaks(perc_df, 30)
ts_df['perc'] = ts_df['perc'].ewm(span=5).mean()
ts_df[13] = ts_df[13]*1.2
ts_df[138] = ts_df[138] * 1.8

ts_df['stationary'] = np.ones(ts_df.shape[0]) + np.sin(2*np.pi*np.arange(0, ts_df.shape[0])/96)*0.001

ts_df = ts_df[['stationary',37, 150, 78, 73, 138, 11, 13, 54, 66, 'slow', 'perc']]

for col in ts_df.columns:
    if col != 'stationary':
        ts_df = taper_end(ts_df, 1447, col, 0.05)

ts_df = taper_at_ind(ts_df, 580, 78, .1)
ts_df = taper_at_ind(ts_df, 500, 73, .1)
ts_df = taper_at_ind(ts_df, 1300, 54, .1)
ts_df = taper_at_ind(ts_df, 600, 37, .1)
ts_df = taper_at_ind(ts_df, 1300, 11, .1)
ts_df = taper_at_ind(ts_df, 1100, 66, .1)
ts_df = taper_at_ind(ts_df, 900, 138, .1)
ts_df = taper_at_ind(ts_df, 600, 150, .1)
ts_df = taper_at_ind(ts_df, 500, 37, .1)

fig, ax = plt.subplots(1,1, figsize=(5,4))


for ind, col in enumerate(ts_df.columns):

    ax.plot(ts_df.index[:1505],ts_df[col].iloc[:1505], zorder=ind, label=col)
    z1 = np.zeros(1505)
    z2 = np.array(ts_df[col].iloc[:1505])
    ax.fill_between(ts_df.index[:1505], z1, z2, zorder=ind, alpha=.5)
    
ax.legend(bbox_to_anchor=(1.0, 1.0))

#%%
end = 3060

s_47 = 1460
x_sin_47 = np.arange(1460, 2900)
y_sin_47 = 20*np.sin(2*np.pi*((x_sin_47-1460)/363) + 0.4*2*np.pi)+47

s_63 = 1470
x_sin_63 = np.arange(s_63, 2900)
y_sin_63 = 20*np.sin(2*np.pi*((x_sin_63-s_63)/370) + 0.70*2*np.pi)+63

s_147 = 2220
x_sin_147 = np.arange(s_147, 2900)
y_sin_147 = 80*np.sin(2*np.pi*((x_sin_147-s_147)/340) + 0.45*2*np.pi)+592

xs = [x_sin_47, x_sin_63, x_sin_147]
ys = [y_sin_47, y_sin_63, y_sin_147]
colors= ['mediumturquoise', 'steelblue', 'cornflowerblue']


plot_stft(x_stft_db, EXPECTED_SAMPLE_RATE, xs, ys, colors, 'sin_waves', [], [], None)
plt.show()


#%%

x_sin_47 = np.arange(1460, 2900)
y_sin_47 = 20*np.sin(2*np.pi*((x_sin_47-1460)/363) + 0.4*2*np.pi)+47

x_sin_63 = np.arange(s_63, 2900)
y_sin_63 = 20*np.sin(2*np.pi*((x_sin_63-s_63)/370) + 0.70*2*np.pi)+63

x_sin_147 = np.arange(s_147, 2900)
y_sin_147 = 28*np.sin(2*np.pi*((x_sin_147-s_147)/363) + 0.45*2*np.pi)+151

high_sin = np.zeros(end)
hs = pd.DataFrame(high_sin)
hs['ewm'] = hs[0][::-1].ewm(span=10).mean()
hs['r_ewm'] = hs['ewm'][::-1]
high_sin[:s_147] = hs['r_ewm'].iloc[:s_147]
high_sin[s_147:2900] = 0.025*np.sin(2*np.pi*((x_sin_147-s_147)/363) + 0.45*2*np.pi)
high_sin = high_sin*15


fig, ax = plt.subplots(1,1, figsize=(6,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.plot(np.arange(0, end), high_sin, color='darkcyan')

ax.set_ylabel('Coefficient')
ax.set_xlabel('Frame (32 fps, 1:35 seconds total)')
#fig.savefig(project_path + 'figures/latent_coeff_high_sin.png', bbox_inches='tight', dpi=300)
 
ts_df['high_sin'] = high_sin

#%%
low_sin = np.zeros(end)
low_sin[s_63:2900] = -0.02
ls = pd.DataFrame(low_sin)
ls['ewm'] = ls[0][::-1].ewm(span=10).mean()
ls['r_ewm'] = ls['ewm'][::-1]
low_sin[:s_63] = ls['r_ewm'].iloc[:s_63]
low_sin[s_63:2900] = 0.025*np.sin(2*np.pi*((x_sin_63-s_63)/370) + 0.70*2*np.pi)
low_sin = low_sin*21

fig, ax = plt.subplots(1,1, figsize=(6,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.plot(np.arange(0, end), low_sin, color='mediumspringgreen')
ax.set_ylabel('Coefficient')
ax.set_xlabel('Frame (32 fps, 1:35 seconds total)')
fig.savefig(project_path + 'figures/latent_coeff_low_sin.png', bbox_inches='tight', dpi=300)

ts_df['low_sin'] = low_sin


#%%
base_sin = np.zeros(end)
bs = pd.DataFrame(base_sin)
bs['ewm'] = bs[0][::-1].ewm(span=10).mean()
bs['r_ewm'] = bs['ewm'][::-1]
base_sin[:s_47] = bs['r_ewm'].iloc[:s_47]
base_sin[s_47:2900] = 0.025*np.sin(2*np.pi*((x_sin_47-1460)/363) + 0.4*2*np.pi)
base_sin = base_sin*21

fig, ax = plt.subplots(1,1, figsize=(6,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.plot(np.arange(0, end), base_sin, color='mediumspringgreen')
ax.set_ylabel('Coefficient')
ax.set_xlabel('Frame (32 fps, 1:35 seconds total)')
fig.savefig(project_path + 'figures/latent_coeff_low_sin.png', bbox_inches='tight', dpi=300)

ts_df['base_sin'] = base_sin

#%%
from librosa.beat import beat_track

def get_ewm_beats(data):
    
    data = data.copy(deep=True)
    _, beats = beat_track(y, sr)
    
    ts_df = pd.DataFrame(index=data.columns, columns=['beats'])
    
    for ind in ts_df.index:
        if ind in list(beats):
            ts_df.loc[ts_df.index==ind, 'beats'] = 1
        else:
            ts_df.loc[ts_df.index==ind, 'beats'] = 0
            
    ts_df['beats'] = ts_df['beats'].astype(float)
    ts_df['beats'] = ts_df['beats'].ewm(span=9).mean()
    ts_df['beats'] = ts_df['beats'][::-1].ewm(span=4).mean()[::-1]
 
    ts_df.plot()
    
    return ts_df['beats']*2.5

ts_df['beats'] = get_ewm_beats(perc_df.iloc[:,:3060])

ts_df['stationary2'] = ts_df['stationary']
#%%


ramp = np.arange(0,1, 1/(1560-1496))
for col in ['stationary2', 'low_sin', 'base_sin', 'high_sin', 'beats']:
 
    for i, ind in enumerate(np.arange(1496, 1560)):
        ts_df.loc[ts_df.index==ind, col] = ts_df.loc[ind, col]*ramp[i]
     
dec = np.arange(1,0, -1/(1560-1496))
for col in ts_df.columns[:11]:
    for i, ind in enumerate(np.arange(1496, 1560)):
        ts_df.loc[ts_df.index==ind, col] = ts_df.loc[ind, col]*dec[i]
        
for ind in ts_df.index:
    for col in ts_df.columns[:11]:
        if ind >= 1560 and ind < 3060:
           ts_df.loc[ts_df.index==ind, col] = 0 
        
for col in ['low_sin', 'base_sin', 'high_sin']:
    
    ts_df = taper_end(ts_df, 2889, col, thresh=0.05, factor=.95)
    
#%%
ts_df = ts_df[['stationary', '37', '150', '78', '73', '138', '11', '13', '54', '66',
       'slow', 'perc', 'stationary2', 'high_sin', 'low_sin', 'base_sin', 'beats']]
       

ts_df.to_csv(project_path  + 'midterm_time_series_latent_coeffs.csv')

#%%
ts_df = ts_df.loc[np.arange(1460,1506), :]
ts_df = ts_df.loc[np.arange(2888,3060), :]


#%%
ts_df = ts_df[['stationary2', 'high_sin', 'low_sin', 'base_sin', 'beats']]
ts_df = ts_df.loc[1460,:]

fig, ax = plt.subplots(1,1, figsize=(5,3))

for ind, col in enumerate(ts_df.columns):

    ax.plot(ts_df.index[:],ts_df[col].iloc[:], zorder=ind, label=col)
    z1 = np.zeros(ts_df.shape[0])
    z2 = np.array(ts_df[col])
    ax.fill_between(ts_df.index, z1, z2, zorder=ind, alpha=.5)
    
ax.legend(bbox_to_anchor=(1.05, 1.05))
#%%
from scipy.signal import find_peaks
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def plot_key(data):
    
    data = data.copy(deep=True)
    data = data + abs(data.min().min())
    
    freq_dt = pd.DataFrame(data.sum(axis=1), columns=['sums'])
        

    fig, ax = plt.subplots(2,2, figsize=(10,6))
    ax[0,0].plot(freq_dt.index, freq_dt['sums'], color='black', alpha=.5)

    # X = freq_dt.index
    # X = np.reshape(X, (len(X), 1))
    # y = freq_dt['sums'].values
    # pf = PolynomialFeatures(degree=5)
    # Xp = pf.fit_transform(X)
    
    # md2 = LinearRegression()
    # md2.fit(Xp, y)
    # trendp = md2.predict(Xp)

    # ax[0,0].plot(X, trendp, color='turquoise')

    # detr = [y[i] - trendp[i] for i in range(0, len(y))]     

    # min_detr = min(detr)
    # detr = [x+np.abs(min_detr) for x in detr]
    
    # ax[0,1].plot(detr)
 
    # indices = find_peaks(detr)[0]
    # peaks = []
    # for ind in indices:
    #     ax[0,1].scatter(ind, detr[ind], color='red', linewidth=0, s=20,alpha=1, zorder=4)
    #     ax[1,0].scatter(ind, detr[ind], color='red', linewidth=0, s=20,alpha=1, zorder=4)
    #     peaks.append(detr[ind])
       
    # peak_df = pd.DataFrame(indices, columns=['indices'])
    # peak_df['peaks'] = peaks
    
plot_key(perc_df.iloc[:30,1440:3200])
#%%

    #indices = find_peaks(detr)[0]



# #%%

# def prompts_to_textfile(ts_df, filename):
    
#     all_prompts = ''
    
#     for t in ts_df.index:
#         all_prompts = all_prompts + ts_df.loc[t, 'full'] + '\n'
    
#     text_file = open(filename, "w")
#     text_file.write(all_prompts)
#     text_file.close()
    
#     return all_prompts
        

# prompt = 'smoke cloud rave, minimalist, wet, oiled, rave lighting, \
# Futuristic lasers tracing, geometry abstract acrylic, colorsmoke, \
# pyramid hoodvisor, raindrops, geomtric, \
# fractal, octane hyperrealism photorealistic airbrush collage painting, \
# monochrome, neon fluorescent colors, minimalist, \
# retro-scifi cave, kaws, mondrian, \
# hannah af klint perfect geometry abstract acrylic, rave, retro glitch art, '

# prompt = 'Futuristic lasers tracing, colorsmoke, pyramid hoodvisor, raindrops, wet, oiled, kaws, mondrian, hannah af klint perfect geometry abstract acrylic, octane hyperrealism photorealistic airbrush collage painting, monochrome, neon fluorescent colors, minimalist, rule of thirds, retro-scifi cave, ' 

# tokens = prompt.split(', ')
# token_dic = {}

# Initialize weighting to 1 at every timestep for each token
# for token in tokens:

#     token_dic[token] = {}      
#     token_dic[token]['coeffs'] = np.ones(ts_mesh.shape[0])
    
# Adjust token weighting given a range and time series of coefficients 

# token_dic['colorsmoke']['coeffs'] = key_freqs[73]['coeffs']
# token_dic['raindrops']['coeffs'] = key_freqs[54]['coeffs']
# token_dic['retro-scifi cave']['coeffs'] = key_freqs[37]['coeffs']*.5
# token_dic['Futuristic lasers tracing']['coeffs'] = key_freqs[22]['coeffs']*2

# token_dic['monochrome']['coeffs'] = key_freqs[138]['coeffs']*.5
# token_dic['neon fluorescent colors']['coeffs'] = key_freqs[66]['coeffs']*1.5

# token_dic['mondrian']['coeffs'] = key_freqs[11]['coeffs']
# token_dic['hannah af klint perfect geometry abstract acrylic']['coeffs'] = key_freqs[90]['coeffs']*2

# token_dic['octane hyperrealism photorealistic airbrush collage painting']['coeffs'] = key_freqs[11]['coeffs']

# token_dic['fractal']['coeffs'] = np.arange(0, .5, .5/ts_mesh.shape[0])
# token_dic['rave']['coeffs'] = np.arange(0, 2, .5/ts_mesh.shape[0])
# token_dic['geometry abstract acrylic']['coeffs'] = np.arange(0, 2, .5/ts_mesh.shape[0])
# token_dic['pyramid hoodvisor']['coeffs'] = np.arange(0, .5, .5/ts_mesh.shape[0])
# token_dic['kaws']['coeffs'] = np.arange(0, .5, .5/ts_mesh.shape[0])
# token_dic['minimalist']['coeffs'] = np.arange(1, .5, -.5/ts_mesh.shape[0])
# token_dic['wet']['coeffs'] = np.arange(0, .5, .5/ts_mesh.shape[0])
# token_dic['oiled']['coeffs']  = np.arange(0, .5, .5/ts_mesh.shape[0])

# token_dic['retro glitch art']['coeffs'] = np.arange(1, 0, -1/ts_mesh.shape[0])

# Compile prompts with specificed token weightings at each timestep
# ts_df = pd.DataFrame(columns=tokens, index=ts_mesh.index)

# for t in ts_df.index:
    
#     prompt_string_t = "'"
    
#     for token in tokens:
    
#         prompt_string_t = prompt_string_t + '(?P<' + token + '>)' + ":" + str(token_dic[token]['coeffs'][t]) + "," 
            
#         ts_df.loc[ts_df.index==t, token] = token_dic[token]['coeffs'][t]
    
#     prompt_string_t = prompt_string_t + "',"
    
#     ts_df.loc[ts_df.index==t, 'full'] = prompt_string_t

# all_prompts = prompts_to_textfile(ts_df.loc[np.arange(0,1400,200)], project_path+file.split('.')[0].split('/')[-1]+'_prompt_series.txt')

# #%%
# ind = 1200
# fig, ax = plt.subplots(1,1, figsize=(10,6))
# ax.bar(x=ts_df.columns[:-1], height=ts_df.iloc[ind,:-1])
# ax.set_xticklabels(ts_df.columns[:-1], rotation=90, size=12)

# #%%


# # Pre-compute a global reference power from the input spectrum
# rp = np.max(np.abs(D))

# fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)

# img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=rp),
#                          y_axis='log', x_axis='time', ax=ax[0])
# ax[0].set(title='Full spectrogram')
# ax[0].label_outer()

# librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_harmonic), ref=rp),
#                          y_axis='log', x_axis='time', ax=ax[1])
# ax[1].set(title='Harmonic spectrogram')
# ax[1].label_outer()

# librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_percussive), ref=rp),
#                          y_axis='log', x_axis='time', ax=ax[2])
# ax[2].set(title='Percussive spectrogram')
# fig.colorbar(img, ax=ax)

# test_d = np.abs(D_percussive)

#ts_df.to_csv(project_path  + 'time_series_latent_coeffs.csv')

#%%
#%%

# from scipy.signal import find_peaks
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression

# def plot_key(harm_df):
#     harm_df = harm_df.copy(deep=True)
#     harm_df = harm_df.mask(harm_df < np.quantile(harmonic,.0))
#     harm_df = harm_df + harm_df.min().min()
#     harm_df.fillna(0, inplace=True)
    
#     freq_dt = pd.DataFrame(harm_df.sum(axis=1), columns=['sums'])
        

#     fig, ax = plt.subplots(2,2, figsize=(10,6))
#     ax[0,0].plot(freq_dt.index, freq_dt['sums'], color='black', alpha=.5)

#     X = freq_dt.index
#     X = np.reshape(X, (len(X), 1))
#     y = freq_dt['sums'].values
#     pf = PolynomialFeatures(degree=5)
#     Xp = pf.fit_transform(X)
    
#     md2 = LinearRegression()
#     md2.fit(Xp, y)
#     trendp = md2.predict(Xp)

#     ax[0,0].plot(X, trendp, color='turquoise')

#     detr = [y[i] - trendp[i] for i in range(0, len(y))]     

#     min_detr = min(detr)
#     detr = [x+np.abs(min_detr) for x in detr]
    
#     ax[0,1].plot(detr)
 
#     indices = find_peaks(detr)[0]
#     peaks = []
#     for ind in indices:
#         ax[0,1].scatter(ind, detr[ind], color='red', linewidth=0, s=20,alpha=1, zorder=4)
#         ax[1,0].scatter(ind, detr[ind], color='red', linewidth=0, s=20,alpha=1, zorder=4)
#         peaks.append(detr[ind])
       
#     peak_df = pd.DataFrame(indices, columns=['indices'])
#     peak_df['peaks'] = peaks
    
#     ax[1,0].plot(detr)
#     indices_peaks = find_peaks(peak_df['peaks'])[0]
#     initial_inds = []
#     peak_o_peaks = []
#     for ind in indices_peaks:
#         ax[1,0].scatter(peak_df['indices'].iloc[ind], peak_df['peaks'].iloc[ind], color='limegreen', linewidth=0, s=10, alpha=1, zorder=5)
#         peak_o_peaks.append(peak_df['peaks'].iloc[ind])
#         initial_inds.append(peak_df['indices'].iloc[ind]) 
      
#     peak_df2 = pd.DataFrame(initial_inds, columns=['indices'])
#     peak_df2['peaks'] = peak_o_peaks
    
#     X = peak_df2['indices'].values.reshape(-1, 1)
#     Y = peak_df2['peaks'].values.reshape(-1, 1) 
#     linear_regressor = LinearRegression()  
#     linear_regressor.fit(X, Y)  
#     Y_pred = linear_regressor.predict(X) 
#     ax[1,0].plot(X, Y_pred, color='limegreen')
    
#     reg = pd.DataFrame(detr)
#     for ind in reg.index:
#         reg.loc[reg.index==ind, 'detrended'] = reg[0].loc[ind] - linear_regressor.coef_*ind
    
#     ax[1,1].plot(reg.index, reg['detrended'])

#     for ind in freq_dt.index:
#         freq_dt.loc[freq_dt.index==ind, 'full_detrended'] = freq_dt['sums'].loc[ind] - linear_regressor.coef_*ind - trendp[ind]
#         freq_dt.loc[freq_dt.index==ind, 'full_detrended_scale'] =  - linear_regressor.coef_*ind - trendp[ind]
            
#     for ind in peak_df.index:
#         p_ind = peak_df['indices'].loc[ind]
#         peak_df.loc[peak_df.index==ind, 'Additive_transform'] = freq_dt['full_detrended_scale'].loc[p_ind]
#         peak_df.loc[peak_df.index==ind, 'transformed'] = freq_dt['sums'].loc[p_ind] + freq_dt['full_detrended_scale'].loc[p_ind]
    
#     peak_df['scaling_factor'] = peak_df['transformed']/peak_df['peaks']

#     # peak_df['ewm'] = peak_df['scaling_factor'].ewm(span=2).mean()
#     # peak_df['r_ewm_scaling'] = peak_df['ewm'][::-1].ewm(span=2).mean()[::-1] 
    
#     ax[1,1].plot(peak_df['indices'], peak_df['scaling_factor']*peak_df['peaks'],color='orange')
    
#     return peak_df

    
# peaks = plot_key(harm_df)

# peaks.to_csv(project_path+file.split('.')[0].split('/')[-1]+'_peaks_and_scaling.csv')
# #%%
# def mel(Hz):
#     mel = 1/np.log(2) * (np.log(1 + (Hz/1000))) * 1000
#     return mel
    
# def get_similarity(data):
    
#     data = data.copy(deep=True)
#     data = data + np.abs(data.min().min())
 
#     freq_dt = pd.DataFrame(data.sum(axis=1), columns=['sums'])
#     fig, ax = plt.subplots(1,1, figsize=(5,3))
#     ax.plot(freq_dt.index, freq_dt['sums'], color='black', alpha=.5)
    
#     indices = find_peaks(freq_dt['sums'])[0]
#     ys = []
#     for ind in indices:
#         ys.append(freq_dt['sums'].loc[ind])
        
#     ax.scatter(indices, ys, color='red', linewidth=0, s=20,alpha=1, zorder=4)
#     ax.set_xlim([0,250])

#     peaks = pd.DataFrame(indices, columns=['indices'])
#     peaks['peaks'] = ys
    
#     data.where(data > 0, 1, inplace=True)
#     #data.where(data <= 0, 0, inplace=True)

#     for ind in peaks.index:
        
#         freq = int(peaks['indices'].loc[ind])
        
#         for j in [2,3,4]:
            
#             har = int(peaks['indices'].loc[ind]*j)
            
#             if har < data.shape[0]:
                
#                 sim = 1 - (np.abs((data.iloc[freq,:] - data.iloc[har,:])).sum())/data.shape[1]
                
#                 peaks.loc[peaks.index==ind, j] = sim
    
#     peaks.set_index(['indices'], inplace=True)
    
#     w = 1
#     for ind in peaks.index:
#         for j in [2,3,4]:
#             if ind in peaks.index:
#                 if j*ind < peaks.index[-1]:
#                     if peaks.loc[ind, j] > .90:
                        
#                         h_range = np.arange(int(ind*j-w), int(ind*j+w+1), 1)
#                         for h in h_range:
#                             if h in peaks.index:
#                                 peaks.drop(h, inplace=True)
   
#     for ind in peaks.index:
#         if ind > 600:
#             peaks.drop(ind, inplace=True)
               
#     return peaks
    
    
# peaks = get_similarity(harm_df.iloc[:,:1440])