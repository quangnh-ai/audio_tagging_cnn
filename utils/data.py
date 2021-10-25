import numpy as np
import time
import torch
import torch.nn as nn
import csv

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

def get_classes_list(classes_path):
    
    with open(classes_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    
    labels = []
    ids = []

    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    
    return ids, labels

