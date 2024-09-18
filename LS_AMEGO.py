import os
import json
import tqdm
import numpy as np
import torch
import torch.nn as nn
from epic_kitchens.hoa import load_detections
from tools.data import FrameDsetSubsampled, FlowDataset
from torchvision import transforms
import argparse
from omegaconf import OmegaConf

class OnlineClusteringTrack:
    def __init__(self, clustering_threshold):
        self.cosine = torch.nn.CosineSimilarity(dim=1)
        self.threshold = clustering_threshold
        self.clusters = {}
        self.track2cluster = {}
    
    def score_track(self, track, features_track):
        if len(self.clusters) == 0:
            return {0: self.threshold - 1}
        scores = {}
        for cluster_id in self.clusters.keys():
            scores[cluster_id] = np.mean([self.cosine(
                features_track[i], features_track[track]).item() for i in self.clusters[cluster_id]])
        return scores
    
    def step(self, track, features_track):
        scores = self.score_track(track, features_track)
        if max(scores.values()) >= self.threshold:
            cluster = max(scores, key=scores.get)
            self.track2cluster[track] = cluster
            self.clusters[cluster].append(track)
        else:
            new_cluster = len(self.clusters)
            self.track2cluster[track] = new_cluster
            self.clusters[new_cluster] = [track]
        return self.track2cluster[track]

class LS_AMEGO:
    def __init__(self, dset, root, config):
        self.dset = dset
        self.root = root
        self.config = config
        
        self.output_dir = config.output_dir
        self.fps = config.fps
        self.v_id = config.v_id
        self.consecutive = config.consecutive
        self.flow_threshold = config.flow_threshold
        self.hand_det_score = config.hand_det_score
        self.no_filter_flow = config.no_filter_flow
        self.no_filter_hands = config.no_filter_hands
        
        self.net = self._initialize_network()
        self.dataset = FrameDsetSubsampled(root, self.fps, self.v_id, dset.name, kwargs={'video_cfg': config.dset_kwargs})
        self.flow_dataset = FlowDataset(root, self.fps, self.v_id, dset.name, kwargs={'video_cfg': config.dset_kwargs})
        self.detections = load_detections(self.dataset.dset.detections_path(self.v_id))

        self.grouped_tracks = []
        self.group = []
        self.final_results = {}
        self.features_track = {}
        
        self.clustering = OnlineClusteringTrack(config.clustering_threshold_ls)

    def _initialize_network(self):
        net = torch.hub.load("facebookresearch/swag", model="vit_l16_in1k")
        net.head = nn.Identity()
        return net

    def step(self, frame_idx):
        frame = self.dataset.frames[frame_idx]
        flow = self._get_flow(frame_idx)
        hands = self._get_hands(frame[1] - 1)
        
        if self._should_process_frame(hands, flow):
            self.group.append(frame[1])
        else:
            self._process_group()
    
    def _get_flow(self, frame_idx):
        if frame_idx == 0:
            return 0
        flow = self.flow_dataset[self.dataset.frames[frame_idx][1] - 1]
        return torch.norm(flow, 2, dim=[0, 1, 2]).item()
    
    def _get_hands(self, frame_num):
        detection = self.detections[frame_num]
        return [hand for hand in detection.hands if hand.score >= self.hand_det_score]
    
    def _should_process_frame(self, hands, flow):
        return (self.no_filter_hands or len(hands) != 0) and (self.no_filter_flow or flow <= self.flow_threshold)
    
    def _process_group(self):
        if len(self.group) >= self.consecutive:
            self.grouped_tracks.append((self.group[0], self.group[-1]))
        self.group = []

    def extract_feat(self, track):
        features = []
        for frame in range(track[1], track[2]):
            image = self.dataset.dset.load_image((self.v_id, frame))
            image = self.dataset.dset.transform(image)
            with torch.no_grad():
                features.append(self.net(image.unsqueeze(0).cuda()))
        return torch.stack(features).mean(0, keepdim=True)

    def process(self):
        for frame_idx in tqdm.tqdm(range(len(self.dataset.frames))):
            self.step(frame_idx)
        
        self._process_group()  # Process any remaining frames in the group
        
        for track_i, track in enumerate(self.grouped_tracks):
            track_num = track_i + 1
            track_boundaries = (track_num, track[0], track[-1])
            self.features_track[track_num] = self.extract_feat(track_boundaries)
            cluster = self.clustering.step(track_num, self.features_track)
            
            self.final_results[track_num] = {
                'cluster': cluster,
                'features': self.features_track[track_num].cpu().numpy().tolist(),
                'num_frame': list(range(track[0], track[-1]))
            }
        
        self._save_results()

    def _save_results(self):
        output_dir = os.path.join(self.output_dir, 'LS_AMEGO')
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, self.v_id + ".json"), 'w') as file:
            json.dump(list(self.final_results.values()), file, indent=4)

def parse_args(config_keys):
    parser = argparse.ArgumentParser(description='Modify configuration parameters.')
    for key in config_keys:
        parser.add_argument(f'--{key}', type=str, help=f'Override {key}')
    return parser.parse_args()

if __name__ == '__main__':
    config = OmegaConf.load('configs/default.yaml')
    config_keys = list(config.keys())
    args = parse_args(config_keys)

    for key in config_keys:
        value = getattr(args, key, None)
        if value is not None:
            if isinstance(config[key], bool):
                value = value.lower() in ['true', '1']
            elif isinstance(config[key], int):
                value = int(value)
            elif isinstance(config[key], float):
                value = float(value)
            config[key] = value

    if config.dset == 'epic':
        from tools.data import EPICDataset
        dset = EPICDataset(config.root)
    else:
        from tools.data import SingleVideoDataset
        dset = SingleVideoDataset(config.root, config.v_id, config)

    processor = LS_AMEGO(dset, config.root, config)
    processor.process()