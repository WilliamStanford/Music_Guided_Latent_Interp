import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import display as librosadisplay

from IPython.display import Audio, Javascript
from pydub import AudioSegment
from scipy.io import wavfile
import seaborn as sns

project_path = '/Users/williamstanford/Desktop/Fall2022/Neural_Rendering/'
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
  
  fig.savefig(project_path + 'figures/latent_spectrogram_'+version+'.png', bbox_inches='tight', dpi=300)
  

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
fig.savefig(project_path + 'figures/latent_spectrogram_base.png', bbox_inches='tight', dpi=300)

#%%
plot_stft(x_stft_db, EXPECTED_SAMPLE_RATE, [], [], [], 'melody', [], [], None)
#%%
end = 1472
# x_sin_47 = np.arange(1460, end)
# y_sin_47 = 20*np.sin(2*np.pi*((x_sin_47-1460)/363) + 0.4*2*np.pi)+47
s_63 = 730
x_sin_63 = np.arange(s_63, 1440)
y_sin_63 = 20*np.sin(2*np.pi*((x_sin_63-s_63)/370) + 0.70*2*np.pi)+63

s_147 = 390
x_sin_147 = np.arange(s_147, 1440)
y_sin_147 = 28*np.sin(2*np.pi*((x_sin_147-s_147)/363) + 0.45*2*np.pi)+151

xs = [x_sin_63, x_sin_147]
ys = [y_sin_63, y_sin_147]
colors= ['mediumturquoise', 'steelblue', 'cornflowerblue']
colors = ['mediumspringgreen', 'darkcyan']

plot_stft(x_stft_db, EXPECTED_SAMPLE_RATE, xs, ys, colors, 'sin_waves', [], [], None)
plt.show()


#%%
high_sin = np.zeros(end)
high_sin[s_147:1440] = -0.02
hs = pd.DataFrame(high_sin)
hs['ewm'] = hs[0][::-1].ewm(span=10).mean()
hs['r_ewm'] = hs['ewm'][::-1]
high_sin[:s_147] = hs['r_ewm'].iloc[:s_147]
high_sin[s_147:1440] = 0.025*np.sin(2*np.pi*((x_sin_147-s_147)/370) + 0.70*2*np.pi)
high_sin = high_sin*18
high_sin[1439:1459] = np.arange(-0.1, 0, 0.005)

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
# Main latent coefficient determination 
end = 1472

spec_df_sm = spec_df.iloc[237:240,:end] + 80
spec_df_sm = spec_df_sm.mean(axis=0)
spec_df_sm = spec_df_sm/spec_df_sm.max()*.5
spec_df_sm = pd.DataFrame(spec_df_sm)

spec_df_sm['ewm'] = spec_df_sm[0].ewm(span=20).mean()
spec_df_sm['r_ewm'] = spec_df_sm['ewm'][::-1].ewm(span=10).mean()[::-1]

fig, ax = plt.subplots(1,1, figsize=(6,3))
ax.plot(list(spec_df_sm.index)[:end], list(spec_df_sm[0])[:end], color='black', linewidth=.5, alpha=.5)
ax.plot(list(spec_df_sm.index)[:end], list(spec_df_sm['r_ewm'])[:end], color='darkslateblue', linewidth=1)
ax.text(580, .005, 'smoothing', fontsize=10, color='black')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Coefficient')
ax.set_xlabel('Frame (32 fps, 45 seconds total)')
fig.savefig(project_path + 'figures/latent_coeff_main.png', bbox_inches='tight', dpi=300)
  
ts_df['Main'].iloc[-31:] = np.arange(-.3,0,.1)

ts_df['Main'] = spec_df_sm['r_ewm']

#%%
from scipy import signal

# Upper melody latent coefficient determination 

spec_df_mel = spec_df.iloc[245:290,:end] + 80
spec_df_mel = spec_df_mel.max(axis=0)
spec_df_mel = spec_df_mel/spec_df_mel.max()
spec_df_mel = pd.DataFrame(spec_df_mel, columns=['max'])
spec_df_mel['ewm'] = spec_df_mel['max'].ewm(span=10).mean()
spec_df_mel['detrend'] = signal.detrend(spec_df_mel['ewm'])*6+.6
spec_df_mel['detrend_ewm'] = spec_df_mel['detrend'].ewm(span=10).mean()
spec_df_mel['detrend_r_ewm'] = spec_df_mel['detrend_ewm'][::-1].ewm(span=10).mean()[::-1]
spec_df_mel['detrend_r_ewm_ramp'] = np.arange(.005, 1, (1-.005)/end)*spec_df_mel['detrend_r_ewm']
spec_df_mel['detrend_r_ewm_ramp'][-30:] = np.arange(0, 1, (1-0)/30)[::-1]*spec_df_mel['detrend_r_ewm_ramp'][-30:]

spec_df_mel['detrend_r_ewm_ramp'] = spec_df_mel['detrend_r_ewm_ramp']*0.4
fig, ax = plt.subplots(1,1, figsize=(6,3))
ax.plot(np.arange(0,end), spec_df_mel['max']*0.4, color='black', linewidth=.5, alpha=.5)
ax.plot(np.arange(0,end), spec_df_mel['detrend_r_ewm_ramp'], color='fuchsia', linewidth=.75)
ax.text(0, 0.3, 'smoothing, detrending,\nlinear ramp + taper', fontsize=10, color='black')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Coefficient')
ax.set_xlabel('Frame (32 fps, 45 seconds total)')
fig.savefig(project_path + 'figures/latent_coeff_back_melody.png', bbox_inches='tight', dpi=300)
  
ts_df['Upper_Melody'] = spec_df_mel['detrend_r_ewm_ramp']

#%%
fig, ax = plt.subplots(1,1, figsize=(6,3))
finale = np.zeros(end)
finale[-32:] = np.sin(2*np.pi*((np.arange(0,32))/64))
ax.plot(np.arange(0,end), finale , color='black', linewidth=.5, alpha=.5)
ts_df['finale'] = finale

#%%
fig, ax = plt.subplots(1,1, figsize=(10,2))
for col, color in zip(ts_df.columns[1:], ['darkcyan', 'mediumspringgreen', 'darkslateblue', 'fuchsia', 'gold']):
    ax.plot(np.arange(0,end), ts_df[col], color=color, linewidth=.75, alpha=.8)
    
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig(project_path + 'figures/latent_coeffs_all.png', bbox_inches='tight', dpi=600)

#%%

ts_df['starting'] = np.ones(ts_df.shape[0])
ts_df['starting'].iloc[-32:-22] = np.arange(1,0,-0.1)
ts_df['starting'].iloc[-22:] = 0

ts_df.to_csv(project_path  + 'time_series_latent_coeffs.csv')
