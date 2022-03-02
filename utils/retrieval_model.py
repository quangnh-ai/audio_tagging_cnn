import h5py
from matplotlib.image import thumbnail
import numpy as np
import json
import argparse

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--mapping_kf2shot', type=str)

    args = parser.parse_args()
    return args

class Retrieval:
    def __init__(self, args):
        self.data = h5py.File(args.features_path, 'r')
        
        content = open(args.mapping_kf2shot)
        self.mapping_dict = json.load(content)

        self.features = self.data.get('features')
        self.ids = self.data.get('shot_ids')
        
        self.ids = [id.decode("utf-8") for id in self.ids]
        self.features = [feature.decode("utf-8") for feature in self.features]

    def retrieval(self, query):

        results = []

        for i in range(len(self.features)):
            if query in self.features[i]:
                # results.append(self.ids[i])
                video_id = self.ids[i].split('_')[0]
                dataset = None
                if int(video_id) <= 7475:
                    dataset = 'V3C1'
                else:
                    dataset = 'V3C2'
                keyframe_ids = self.ids[i]
                # shot_id = self.mapping_dict.get(keyframe_ids)
                thumbnail_path = dataset + '/' + video_id + '/' + keyframe_ids + '.png'
                results.append({
                    "dataset": dataset,
                    "video_id": video_id,
                    "shot_id": keyframe_ids,
                    # "frame_id": keyframe_ids,
                    'thumbnail_path': thumbnail_path
                })
        return results