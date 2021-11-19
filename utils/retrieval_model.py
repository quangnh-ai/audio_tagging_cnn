import h5py
import numpy as np

import argparse

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str)

    args = parser.parse_args()
    return args

class Retrieval:
    def __init__(self, args):
        self.data = h5py.File(args.features_path, 'r')

        self.features = self.data.get('features')
        self.ids = self.data.get('shot_ids')
        
        self.ids = [id.decode("utf-8") for id in self.ids]
        self.features = [feature.decode("utf-8") for feature in self.features]

    def retrieval(self, query):

        results = []

        for i in range(len(self.features)):
            if query in self.features[i]:
                results.append(self.ids[i])
        
        return results