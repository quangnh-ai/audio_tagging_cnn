import h5py
from matplotlib.image import thumbnail
import numpy as np
import json
import argparse

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str)
    # parser.add_argument('--mapping_kf2shot', type=str)

    args = parser.parse_args()
    return args

class Retrieval:
    def __init__(self, args):
        self.data = h5py.File(args.features_path, 'r')
        
        # content = open(args.mapping_kf2shot)
        # self.mapping_dict = json.load(content)

        # self.mapping_keys = list(self.mapping_dict.keys())
        # self.mapping_values = list(self.mapping_dict.values())

        self.features = self.data.get('features')
        self.shot_ids = self.data.get('shot_ids')
        self.keyframe_ids = self.data.get('keyframe_ids')
        
        self.ids = [id.decode("utf-8") for id in self.ids]
        self.features = [feature.decode("utf-8") for feature in self.features]
    
    def get_keyframe_id(self, shot_id):
        idx = self.mapping_values.index(shot_id)
        result = self.mapping_keys[idx]
        return result    

    def retrieval(self, query):

        results = []

        for i in range(len(self.features)):
            if query in self.features[i]:
                # results.append(self.ids[i])
                video_id = self.shot_ids[i].split('_')[0]
                dataset = None
                if int(video_id) <= 7475:
                    dataset = 'V3C1'
                else:
                    dataset = 'V3C2'
                shot_id = self.shot_ids[i].split('_')[1]
                keyframe_id = self.keyframe_ids[i].split('_')[1]
                # keyframe_id = self.get_keyframe_id(shot_id)
                thumbnail_path = dataset + '/' + video_id + '/' + shot_id + '/' + video_id + '_' + shot_id + '_' + keyframe_id + '.jpg'
                results.append({
                    "dataset": dataset,
                    "video_id": video_id,
                    "shot_id": int(shot_id),
                    "frame_id": int(keyframe_id),
                    'thumbnail_path': thumbnail_path
                })
        return results