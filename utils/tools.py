import os
import numpy as np
import librosa
import torch
import csv

from data import move_data_to_device

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
            result = result + labels[i].lower() + " "
    
    return result
    
