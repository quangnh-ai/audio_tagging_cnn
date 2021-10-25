import os
import sys
import wave
import numpy as np
import argparse
import librosa
import torch
import csv

from utils.model import Cnn14
from utils.data import move_data_to_device, get_classes_list

from configs.config import init_config

my_cfg = init_config()

def get_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--audio_path', type=str, required=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = get_arg()

    sample_rate = int(my_cfg['model']['sample_rate'])
    window_size = int(my_cfg['model']['window_size'])
    hop_size = int(my_cfg['model']['hop_size'])
    mel_bins = int(my_cfg['model']['mel_bins'])
    fmin = int(my_cfg['model']['fmin'])
    fmax = int(my_cfg['model']['fmax'])
    classes_num = int(my_cfg['model']['classes_num'])

    print('(+) Init Model')

    model = Cnn14(sample_rate=sample_rate, window_size=window_size, hop_size=hop_size,
                  mel_bins=mel_bins, fmin=fmin, fmax=fmax, classes_num=classes_num)

    print('(+) Get classes list')
    classes_path = my_cfg['data']['classes_path']

    ids, labels = get_classes_list(classes_path)

    checkpoint_path = args.checkpoint_path
    print('(+) Load checkpoint')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.to('cuda')
    print('(+) GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    print('(+) Audio Tagging')
    audio_path = args.audio_path
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = waveform[None, :]
    waveform = move_data_to_device(waveform, 'cuda')

    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)
    
    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    for k in range(20):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))

    