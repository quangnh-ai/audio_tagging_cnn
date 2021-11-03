import os
import numpy as np
import librosa
from scipy.signal import waveforms
import torch
import csv

from utils.data import move_data_to_device

def audio_tagging(model, audio_path, labels, sample_rate=32000):
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = waveform[None, :]
    waveform = move_data_to_device(waveform, 'cuda')

    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)
    
    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    sorted_indexes = np.argsort(clipwise_output)[::-1]
    
    result = ""
    for i in sorted_indexes:
        if clipwise_output[i] >= 0.08:
            result = result + labels[i] + " "
    
    return result
    
def tagging_on_folder(model, data_path, labels):
    audios_list = os.listdir(data_path)
    
    results = []
    shot_ids = []

    audio_count = len(audios_list)
    count = 0

    for audio in audios_list:
        audio_path = os.path.join(data_path, audio)

        audio_shots = os.listdir(audio_path)
        audio_shots = sorted(audio_shots)

        for audio_shot in audio_shots:
            audio_shot_path = os.path.join(audio_path, audio_shot)
            (waveform, _) = librosa.core.load(audio_shot_path, sr=32000, mono=True)
            waveform = waveform[None, :]
            waveform = move_data_to_device(waveform, 'cuda')

            with torch.no_grad():
                model.eval()
                try:
                    batch_output_dict = model(waveform, None)
                except RuntimeError:
                    continue
            
            clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
            sorted_indexes = np.argsort(clipwise_output)[::-1]

            result = ""

            for i in sorted_indexes:
                if clipwise_output[i] >= 0.08:
                    result = result + labels[i] + " "
            
            if result != "":
                results.append(result)
                shot_ids.append(audio_shot)

        count += 1
        print("(+) Process: ", 100 * count/audio_count , '%')
    
    return shot_ids, results
