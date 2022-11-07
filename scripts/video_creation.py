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
audio_frames = 604314


#%%

#%%
import cv2
import os
import glob


image_folder = project_path + 'interp/*'
video_name = project_path + 'midterm_interp.avi'

images = glob.glob(image_folder)
images.sort()

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 64, (width,height))

for i, image in enumerate(images):
    video.write(cv2.imread(os.path.join(image_folder, image)))
    
    if i%200 == 0:
        print(i)

cv2.destroyAllWindows()
video.release()


#%%
import os

project_path = '/Users/williamstanford/Desktop/Fall2022/Neural_Rendering/'


os.system("ffmpeg -y -ss 0 -t 01:36 -i "+project_path+"audio/RichDivKid.mp3 "+project_path+"audio/RichDivKid_clipped.mp3")
os.system("ffmpeg -y -i "+project_path+"audio/RichDivKid_clipped.mp3 -af 'afade=t=out:st=1:33:d=3' "+project_path+"audio/RichDivKid_clipped_fade.mp3")
os.system("ffmpeg -y -i "+project_path+"midterm_interp.avi -i "+project_path+"audio/RichDivKid_clipped.mp3 -c copy -map 0:v:0 -map 1:a:0 "+project_path+"midterm_w_audio.avi")

#os.system("ffmpeg -y -i "+project_path+"output.avi -vf 'fade=t=in:st=0:d=2,fade=t=out:st=45:d=2' -c:a copy "+project_path+"output_fade.avi")

#os.system("ffmpeg -y -i "+project_path+"output.avi  "+project_path+"output.mp4")
















