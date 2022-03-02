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
    
    def get_keyframe_id(self, shot_id):
        for keyframe_id, shot_idx in self.mapping_dict.items():
            if shot_id == shot_idx:
                return keyframe_id
        return ''    

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
                shot_id = self.ids[i]
                keyframe_id = self.get_keyframe_id(shot_id)
                thumbnail_path = dataset + '/' + video_id + '/' + shot_id + '/' + video_id + '_' + shot_id + '_' + keyframe_id + '.jpg'
                results.append({
                    "dataset": dataset,
                    "video_id": video_id,
                    "shot_id": shot_id,
                    "frame_id": keyframe_id,
                    'thumbnail_path': thumbnail_path
                })
        return results