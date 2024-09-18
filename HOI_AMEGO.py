import numpy as np
import cv2 as cv
import os
import sys
import torch
import torch.nn as nn
from tools.data import EPICDataset, SingleVideoDataset
from tools.data import ObjectFrameDsetSubsampled
import tqdm
import json
import torch
import pandas as pd
import ast
from PIL import ImageDraw
import numpy as np
from omegaconf import OmegaConf
import argparse

prj_path = os.path.abspath(os.path.join('submodules', 'EgoTracks')) 
if prj_path not in sys.path:
    sys.path.insert(0, prj_path)
    
from tracking.config import stark_defaults
from tracking.models.stark_tracker import stark_tracker
            
class OnlineClusteringTrack:
    def __init__(self, clustering_threshold, device) -> None:
                
        self.cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.threshold = clustering_threshold
        self.clusters = {}
        self.track2cluster = {} 
        self.device = device
    
    def score_track(self, track_id, tracks, final_tracks):
        if len(self.clusters) == 0:
            return {0: self.threshold - 1}
        scores = {}
        current_track = tracks[track_id]
        for cluster_id in self.clusters.keys():
            if scores.get(cluster_id, None) is None:
                scores[cluster_id] = np.mean([self.compute_similarity(final_tracks[i]['features'], current_track['features']) for i in self.clusters[cluster_id]])
        return scores
    
    def compute_similarity(self, track_x, track_y):
        return self.cosine(track_x, track_y)[0]
                  
    def step(self, track_id, tracks, final_tracks, prev_track=None, prev_track_conf=0):
        """
        prev_track: the previous cluster that was assigned to the same tracker
        prev_track_conf: the confidence of tracker in detecting the object
        """ 
        scores = self.score_track(track_id, tracks, final_tracks)
        if max(scores.values()) >= self.threshold:
            if prev_track is None or max(scores.values()) > prev_track_conf:    
                cluster = max(scores, key=scores.get)
                self.track2cluster[track_id] = cluster
                self.clusters[cluster].append(track_id)
            else:
                cluster = self.track2cluster[prev_track]
                self.track2cluster[track_id] = cluster
                self.clusters[cluster].append(track_id)
        else:
            if prev_track is None or (1 - max(scores.values())) > prev_track_conf:
                cluster = len(self.clusters)
                self.track2cluster[track_id] = cluster
                self.clusters[cluster] = [track_id]
            else:
                cluster = self.track2cluster[prev_track]
                self.track2cluster[track_id] = cluster
                self.clusters[cluster].append(track_id)
        return cluster
    
    def assign(self, track_id, cluster):
        self.track2cluster[track_id] = cluster
        self.clusters[cluster].append(track_id)

class TrackManager:
    """
    A class to manage tracks and track clusters for a given video.
    -----------
    hoi_iou_threshold : float
        The IOU threshold for human-object interaction bounding boxes to consider them consecutive.
    tracker_hoi_iou : float
        The IOU threshold for tracker detections to be matched to an existing track.
    detection_score : float
        The minimum score for an HOI object detection to be considered valid.
    num_steps : int
        The number of steps in a window to consider a track active.
    window_dim : int
        The dimension of the window in which num_steps detections should be observed.
        i.e. if num_steps detections with IOU > hoi_iou_threshold (wrt the last detection of the track) are observed in 
        the last window_dim frames, the track is active.
    num_steps_no_object : int
        The number of consecutive steps without a detection with IOU > hoi_iou_threshold (wrt the last detection of the track)
        to consider a track inactive.
    tracker_confidence_detection : float
        The minimum confidence of a tracker to consider its detection valid, i.e. to consider it is correctly 
        detecting the initial object.
    iou_nms_tracker : float
        The IOU threshold for non-maximum suppression of trackers, i.e. if two trackers have an IOU > iou_nms_tracker 
        and both have confidence > tracker_nms_confidence, the one with the oldest tracker is kept.
    tracker_nms_confidence : float
        The minimum confidence of a tracker to be considered for non-maximum suppression
    root : str
        The root directory of EPIC Kitchens dataset.
    v_id : str
        The ID of the video.
    total_frames : int
        The total number of frames in the video.
    consider_hand_presence : bool
        Whether to consider the presence of a hand as a trigger to increase num_steps_no_object.
    """
    def __init__(self, dset, root, config):
        self.hoi_iou_threshold = config.hoi_iou_threshold
        self.tracker_hoi_iou = config.tracker_hoi_iou
        self.detection_score = 0.1
        self.num_steps = config.num_steps
        self.window_dim = config.window_dim
        self.num_steps_no_object = config.num_steps_no_object
        self.tracker_confidence_detection = 0.7
        self.iou_nms_tracker = config.iou_nms_tracker
        self.tracker_nms_confidence = 0.4
        self.max_track_id = 0
        self.max_tracker_id = 0
        self.tracks = {}
        self.tracked_tracks = {}
        self.final_tracks = {}
        self.root = root
        self.v_id = config.v_id
        self.p_id = config.v_id.split('_')[0]
        self.total_frames = config.total_frames
        self.consider_hand_presence = config.consider_hand_presence
        self.tracker_usage = config.tracker_usage
        
        net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.net = nn.DataParallel(net)
        self.net.eval().cuda()
        self.device = 'cuda'
        self.dset = dset
        
        self.clustering = OnlineClusteringTrack(clustering_threshold=config.clustering_threshold_hoi, device=self.device)
        self.fps = self.dset.video_fps[config.v_id]
        self.output_dir = config.output_dir

            
    def step(self, frame, left_object_bboxes, right_object_bboxes, left_hand, right_hand, hoi_score_left, hoi_score_right, num_frame):
        save_track = []
        if self.tracker_usage:
            for track in self.tracked_tracks.values():
                # first step is to update all the trackers running
                img = torch.from_numpy(np.array(frame)).permute(2, 0, 1).unsqueeze(0)
                out = track['tracker'].run_model(img)
                assert len(out) == 1, "Only support single object tracking!"
                track['tracker'].update_tracker(out)
                out = out[0]
                track["confidence"].append(out['score'])
                track["obj_bbox"].append(out["bbox"])
                track["num_frame"].append(num_frame)
            
            # secondly we filter the trackers that are overlapping to avoid redundancy
            self._filter_trackers()
             
        # thirdly we check if the object detections are matching with any of the existing tracks, create a new track or update an existing one   
        if len(left_object_bboxes) == 0 or hoi_score_left < self.detection_score:
            matching_track_id_left = -1
        else:
            matching_track_id_left = self._find_matching_track(left_object_bboxes, self.tracks, self.hoi_iou_threshold)
            
            if matching_track_id_left is None:
                matching_track_id_left = self._create_track(frame, left_object_bboxes, num_frame, 'left', hoi_score_left)
            else:
                self._update_track(frame, matching_track_id_left, left_object_bboxes, num_frame, 'left', hoi_score_left)
        
        if len(right_object_bboxes) == 0 or hoi_score_right < self.detection_score:
            matching_track_id_right = -1
        else:
            matching_track_id_right = self._find_matching_track(right_object_bboxes, self.tracks, self.hoi_iou_threshold)
            
            if matching_track_id_right is None:
                matching_track_id_right = self._create_track(frame, right_object_bboxes, num_frame, 'right', hoi_score_right)
            else:
                if matching_track_id_left == matching_track_id_right:
                    side = 'both'
                else:
                    side = 'right'
                self._update_track(frame, matching_track_id_right, right_object_bboxes, num_frame, side, hoi_score_right)        

        if self.tracker_usage:
            trackers_to_update = []
            for tracked_track_id, tracked_track in self.tracked_tracks.items():
                # if tracker has just been created then skip it
                if len(tracked_track['confidence']) == 1:
                    continue
                
                active_tracks = [track_id for track_id, track in self.tracks.items() if track['active']]
                if tracked_track['track_id'][-1] not in active_tracks and tracked_track['confidence'][-1] > self.tracker_confidence_detection:
                    # if tracker is not tracking an active track, then check if it is tracking a track that has become active
                    # i.e. it started to track a certain object which became inactive and now it is active again
                    matched_track = self._find_matching_track(tracked_track['obj_bbox'][-1], self.tracks, self.tracker_hoi_iou, just_active=True)
                    if matched_track is not None and tracked_track['confidence'][-1] > self.tracker_confidence_detection:
                        tracked_track['track_id'].append(matched_track)
                        trackers_to_update.append(tracked_track_id)
                    else:
                        # if tracker is not tracking an active track and it is not tracking a track that has become active, 
                        # then it is not following any track, so we just keep the last track_id
                        tracked_track['track_id'].append(tracked_track['track_id'][-1])
                else:
                    # if tracker is tracking an active track with a sufficient confidence, then it is still following it
                    tracked_track['track_id'].append(tracked_track['track_id'][-1])
                    trackers_to_update.append(tracked_track_id)
                    
            for tracked_track_id in trackers_to_update:
                self._update_track_tracker_pred(frame, tracked_track_id, self.tracked_tracks[tracked_track_id], matching_track_id_left, matching_track_id_right, num_frame)
        
        for track_id in list(self.tracks):
            track = self.tracks[track_id]
            if track_id not in [matching_track_id_left, matching_track_id_right]:
                if not track['active']:
                    track["window_steps"].append(False)
                    track["window_steps"] = track["window_steps"][1:]
                    if sum(track["window_steps"]) == 0:
                        del self.tracks[track_id]
                else:
                    if self.consider_hand_presence:
                        if (('left' in track['side'][-1] and left_hand) or ('right' in track['side'][-1] and right_hand)):
                            track['num_steps_no_object'] += 1
                            if track['num_steps_no_object'] == self.num_steps_no_object:
                                save_track.append(track_id)
                    else:
                        track['num_steps_no_object'] += 1
                        if track['num_steps_no_object'] == self.num_steps_no_object:
                            save_track.append(track_id)
                    
        last_step = num_frame == self.total_frames   

        if last_step:
            for track_id in list(self.tracks):
                track = self.tracks[track_id]
                if track['active']:
                    save_track.append(track_id)
                    
            save_track = list(set(save_track))
            
        self._update_features(frame, num_frame)
        
        for track_id in save_track:
            self._save_track(track_id, num_frame)
        
        if last_step:
            print('Saving...')
            for track_id in self.final_tracks.keys():
                self.final_tracks[track_id]['features'] = self.final_tracks[track_id]['features'].numpy().tolist()
                for i_f, f in enumerate(self.final_tracks[track_id]['all_features']):
                    self.final_tracks[track_id]['all_features'][i_f] = f.numpy().tolist()[0]
            os.makedirs(os.path.join(self.output_dir, 'HOI_AMEGO'), exist_ok=True)
            with open(os.path.join(self.output_dir, 'HOI_AMEGO', self.v_id + ".json"), 'w') as file:
                json.dump(list(self.final_tracks.values()), file, indent=4)
                
    def _tracker_track(self, track_id):
        trackers_track_id = [tracker_id for tracker_id, tracker in self.tracked_tracks.items() if tracker['track_id'][-1] == track_id]
        if len(trackers_track_id) == 0:
            return None
        max_confidence = max([self.tracked_tracks[tracker_id]['confidence'][-1] for tracker_id in trackers_track_id])
        
        if np.isnan(max_confidence):
            return None 
        tracker_track_id = [tracker_id for tracker_id in trackers_track_id if self.tracked_tracks[tracker_id]['confidence'][-1] == max_confidence]
        return tracker_track_id[0]
    
    def _update_track_tracker_pred(self, frame, tracked_track_id, tracked_track, matching_track_id_left, matching_track_id_right, num_frame):
        # first see if current tracker for the track is confident enough, in case update with it
        try:
            current_tracker_id_track = self.tracks[tracked_track['track_id'][-1]]['current_tracker_id_track']
        except:
            # if the tracker is tracking an inactive track, then cannot update any track
            return
        if current_tracker_id_track in self.tracked_tracks and \
            self.tracked_tracks[current_tracker_id_track]['confidence'][-1] > self.tracker_confidence_detection:
            if tracked_track_id != current_tracker_id_track:
                return
            else:
                if num_frame in self.tracks[tracked_track['track_id'][-1]]['num_frame']:
                    # if confidence for current tracker is high enough, then update the track with the current tracker's prediction
                    self.tracks[tracked_track['track_id'][-1]]['obj_bbox'][-1] = tracked_track['obj_bbox'][-1]
                    self.tracks[tracked_track['track_id'][-1]]['score'][-1] = tracked_track['confidence'][-1]
                else:
                    # if the tracker is confident enough and there is no bounding box for the current frame in the track, then add it
                    self.tracks[tracked_track['track_id'][-1]]['obj_bbox'].append(tracked_track['obj_bbox'][-1])
                    self.tracks[tracked_track['track_id'][-1]]['num_frame'].append(num_frame)
                    self.tracks[tracked_track['track_id'][-1]]['score'].append(tracked_track['confidence'][-1])
                    self.tracks[tracked_track['track_id'][-1]]['side'].append(self.tracks[tracked_track['track_id'][-1]]['side'][-1])

        # if current tracker is not confident enough, then look for the tracker with the highest confidence that is tracking the same track
        if tracked_track['confidence'][-1] > self.tracker_confidence_detection:
            
            tracker_id_tracks = [tracker_id for tracker_id in self.tracked_tracks if self.tracked_tracks[tracker_id]['track_id'][-1] == tracked_track['track_id'][-1]]
            tracker_id_track = max(tracker_id_tracks, key=lambda x: self.tracked_tracks[x]['confidence'][-1])
            
            if tracker_id_track != tracked_track_id:
                return
            else:
                self.tracks[tracked_track['track_id'][-1]]['current_tracker_id_track'] = tracker_id_track
                # check if a bounding box is already present for the current frame in the track
                if num_frame in self.tracks[tracked_track['track_id'][-1]]['num_frame']:
                    self.tracks[tracked_track['track_id'][-1]]['obj_bbox'][-1] = tracked_track['obj_bbox'][-1]
                    self.tracks[tracked_track['track_id'][-1]]['score'][-1] = tracked_track['confidence'][-1]
                else:
                    # if the tracker is confident enough and there is no bounding box for the current frame in the track, then add it
                    self.tracks[tracked_track['track_id'][-1]]['obj_bbox'].append(tracked_track['obj_bbox'][-1])
                    self.tracks[tracked_track['track_id'][-1]]['num_frame'].append(num_frame)
                    self.tracks[tracked_track['track_id'][-1]]['score'].append(tracked_track['confidence'][-1])
                    self.tracks[tracked_track['track_id'][-1]]['side'].append(self.tracks[tracked_track['track_id'][-1]]['side'][-1])
            
    def _filter_trackers(self):
        tracked_filtered = {}
        tracked_tracks_id = list(self.tracked_tracks.keys())
        del_track = []
        # scroll trackers backwards so to keep the oldest one
        for i in range(len(self.tracked_tracks) - 1, -1, -1):
            if i in del_track:
                continue
            track_id_i = tracked_tracks_id[i]
            overlapping_trackers = []
            for j in range(i-1, -1, -1):
                if j in del_track:
                    continue
                track_id_j = tracked_tracks_id[j]
                # if the two trackers are overlapping and both have a confidence > tracker_nms_confidence, then keep the oldest one
                iou_value = self._calculate_iou(self.tracked_tracks[track_id_i]['obj_bbox'][-1], self.tracked_tracks[track_id_j]['obj_bbox'][-1])
                if iou_value > self.iou_nms_tracker and self.tracked_tracks[track_id_i]['confidence'][-1] > self.tracker_nms_confidence and self.tracked_tracks[track_id_j]['confidence'][-1] > self.tracker_nms_confidence:
                    overlapping_trackers.append(track_id_j)
            if len(overlapping_trackers) == 0: 
                tracked_filtered[track_id_i] = self.tracked_tracks[track_id_i]
            else:
                # if at least one of the overlapping trackers has a confidence > 0.5, then delete the newest one otherwise delete 
                # all the old ones with confidence > tracker_nms_confidence
                max_conf_over = max([self.tracked_tracks[overl]['confidence'][-1] for overl in overlapping_trackers])
                if max_conf_over > 0.5:
                    del_track.append(track_id_i)      
                else:
                    tracked_filtered[track_id_i] = self.tracked_tracks[track_id_i]
                    for overl_track in overlapping_trackers:
                        del_track.append(overl_track)
        self.tracked_tracks = tracked_filtered
        
            
    def _find_prev_track_tracker(self, track_id):
        
        trackers_on_track = [self.tracked_tracks[tr] for tr in self.tracked_tracks if self.tracked_tracks[tr]['track_id'][-1] == track_id]
        
        def prev_track(tracks, confidences, min_conf=0.5):
            if len(tracks) == 1:
                return None
            last_value = tracks[-1]
            for i_v, value in enumerate(reversed(tracks[:-1])):
                if value != last_value and confidences[len(tracks) - i_v - 1] > min_conf:
                    return value
            return None
                
        prev_values = [prev_track(tr['track_id'], tr['confidence']) for tr in trackers_on_track]
        max_conf = 0
        max_track_prev_object = None
        for track, prev_val in zip(trackers_on_track, prev_values):
            if prev_val is not None and track['confidence'][-1] > max_conf:
                max_conf = track['confidence'][-1]
                max_track_prev_object = prev_val
        return max_track_prev_object, max_conf
    
    def _find_matching_track(self, obj_bbox, tracks, iou_threshold, just_active=False):
        # find the track with the highest IOU with the current detection
        # just_active: if True, only consider active tracks 
        max_iou = 0
        matching_track_id = None
        for track_id, track in tracks.items():
            if just_active and not track["active"]:
                continue
            iou = self._calculate_iou(track["obj_bbox"][-1], obj_bbox)
            if iou > max_iou:
                max_iou = iou
                matching_track_id = track_id
        if max_iou < iou_threshold:
            return None
        return matching_track_id

    def _create_track(self, frame, obj_bbox, frame_num, side, score):
        # create a new track
        track_id = self._get_next_track_id()
        initial_window = [False] * self.window_dim
        initial_window[-1] = True
        self.tracks[track_id] = {
            "track_id": track_id,
            "obj_bbox": [obj_bbox],
            "num_frame": [frame_num],
            "window_steps": initial_window,
            "num_steps_no_object": 0,
            "active": False,
            "all_features": [],
            "side": [side],
            "score": [score]
        }
        
        if sum(initial_window) >= self.num_steps:
            self.tracks[track_id]["active"] = True
            if self.tracker_usage:
                tracker_id =  self._create_tracker_track(frame, obj_bbox, frame_num, track_id)
                self.tracks[track_id]['current_tracker_id_track'] = tracker_id
            return track_id
        
        return track_id
    
    def _create_tracker_track(self, frame, obj_bbox, frame_num, track_id):
        tracker_id = self._get_next_tracker_id()
        self.tracked_tracks[tracker_id] = {
            "track_id": [track_id],
            "obj_bbox": [obj_bbox],
            "num_frame": [frame_num],
            "tracker": self._initialize_tracker(frame, obj_bbox),
            "confidence": [1.0]
        }
        return tracker_id
    
    def _save_track(self, track_id, num_frame):
                
        overlap_track = self._overlapping_tracks(track_id)
        track = self.tracks[track_id]
        
        if len(track['num_frame']) == 0:
            del self.tracks[track_id]
            return
        
        assert (all(len(track[key]) == len(track['all_features']) for key in ['num_frame', 'obj_bbox', 'score', 'side']))
        
        track['features'] = sum(torch.stack(track['all_features'], dim=0)) / len(track['all_features'])
        track['last_frame'] = num_frame - 1
        if overlap_track is None:
            max_track_prev_object, max_conf = self._find_prev_track_tracker(track_id)
            cluster = self.clustering.step(track_id, self.tracks, self.final_tracks, prev_track=max_track_prev_object, prev_track_conf=max_conf)
        else:
            cluster = self.final_tracks[overlap_track]['cluster']
            self.clustering.assign(track_id, cluster)
            
        track['cluster'] = cluster
        output = {key: track[key] for key in ['track_id', 'obj_bbox', 'num_frame', 'all_features', 'features', 'cluster', 'last_frame', 'side']}
        self.final_tracks[track_id] = output
        del self.tracks[track_id]
        
    def _overlapping_tracks(self, track_id):
        """
        function to check if a track overlaps with any of the final tracks, in case this deletes the overlapping part from the track and 
        forces the track to be assigned to the same cluster as the overlapping track
        """
        track = self.tracks[track_id]
        for fin_track_id, fin_track in self.final_tracks.items():
            count = 0
            common_frame = []
            ious = []
            for frame in fin_track['num_frame']:
                if frame in track['num_frame']:
                    ious.append(self._calculate_iou(track['obj_bbox'][track['num_frame'].index(frame)], fin_track['obj_bbox'][fin_track['num_frame'].index(frame)]))
                    count += 1
                    common_frame.append(frame)
            if count > 0:
                if sum(ious) / count > 0.4 or max(ious) > 0.6:
                    frame_idx = [track['num_frame'].index(cmmn_frame) for cmmn_frame in common_frame]
                    for frm_remove in reversed(sorted(frame_idx)):
                        del track['score'][frm_remove]
                        del track['num_frame'][frm_remove]
                        del track['obj_bbox'][frm_remove]
                        del track['all_features'][frm_remove]
                        del track['side'][frm_remove]
                    return fin_track_id
        return None     
    
    def _update_features(self, frame, current_frame):
        for track_id in self.tracks:
            track = self.tracks[track_id]
            if current_frame in track['num_frame'] and len(track['all_features']) < len(track['num_frame']):
                bbox = track['obj_bbox'][track['num_frame'].index(current_frame)]
                frame_cropped = frame.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                frame_cropped = self.dset.transform(frame_cropped)
                features = self.net(frame_cropped.unsqueeze(0).cuda())
                features = features.detach().cpu()
                self.tracks[track_id]['all_features'].append(features)

    def _update_track(self, frame, track_id, obj_bbox, frame_num, side, score):
        
        track = self.tracks[track_id]
        if side == 'both':
            # if side is both it means that it was already updated with the right hand, so just update the hand field
            track['side'][-1] = ('left', 'right')
            return 
        track['side'].append(side)
        track['obj_bbox'].append(obj_bbox)
        track['num_frame'].append(frame_num)
        track['score'].append(score)
        
        # if state is active, then just reset the counter for no contact steps
        if track['active']:
            track['num_steps_no_object'] = 0
            return 
        
        # otherwise check if the track becomes active, i.e. if there are >= num_steps detections in the last window_dim frames
        # in case the track becomes active, create a tracker for it  
        track["window_steps"].append(True)
        track["window_steps"] = track["window_steps"][1:]
                    
        if sum(track["window_steps"]) >= self.num_steps:
            track["active"] = True
            if self.tracker_usage:
                tracker_id =  self._create_tracker_track(frame, obj_bbox, frame_num, track_id)
                track['current_tracker_id_track'] = tracker_id
            return 
        return 
                            
    def _calculate_iou(self, bbox1, bbox2):
        """
        function to calculate the intersection over union between two bounding boxes
        bounding boxes are expected to be in the format [x, y, w, h]
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        area1 = w1 * h1
        area2 = w2 * h2

        intersection_x = max(x1, x2)
        intersection_y = max(y1, y2)
        intersection_w = max(0, min(x1 + w1, x2 + w2) - intersection_x)
        intersection_h = max(0, min(y1 + h1, y2 + h2) - intersection_y)
        intersection_area = intersection_w * intersection_h

        union_area = area1 + area2 - intersection_area

        iou = intersection_area / union_area
        return iou

    def _get_next_track_id(self):
        self.max_track_id += 1
        return self.max_track_id
    
    def _get_next_tracker_id(self):
        self.max_tracker_id += 1
        return self.max_tracker_id
    
    def _initialize_tracker(self, frame, obj_bbox):
        param_module = stark_defaults
        params = param_module.cfg.clone()
        params.MODEL.WEIGHTS = "model_checkpoints/STARKST_ep0001.pth.tar"
        
        tracker_module = stark_tracker
        tracker_class = tracker_module.STARKTracker(params, device=self.device)

        tracker = tracker_class.eval()
        
        meta_data = {
            'target_bbox': obj_bbox,
            'target_id': 0,
        }
                
        image = torch.from_numpy(np.array(frame)).permute(2, 0, 1).unsqueeze(0)
        
        tracker.init_tracker(image, meta_data)
        return tracker
        
def bbox_xyxy_2_bbox_xywh(bbox):
    return [bbox[0][0], bbox[0][1], bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1]]

def parse_args(config_keys):
    parser = argparse.ArgumentParser(
        description='Modify configuration parameters.')

    # Add arguments based on config keys
    for key in config_keys:
        parser.add_argument(f'--{key}', type=str, help=f'Override {key}')

    return parser.parse_args()


if __name__ == '__main__':
    config = OmegaConf.load('configs/default.yaml')

    # Get the keys of the configuration
    config_keys = list(config.keys())

    # Parse CLI arguments
    args = parse_args(config_keys)

    # Update the configuration with CLI arguments
    for key in config_keys:
        value = getattr(args, key, None)
        if value is not None:
            # Convert value to the correct type based on original config type
            if isinstance(config[key], bool):
                value = value.lower() in ['true', '1']
            elif isinstance(config[key], int):
                value = int(value)
            elif isinstance(config[key], float):
                value = float(value)
            config[key] = value

        
    dataset = ObjectFrameDsetSubsampled(config.root, config.fps, config.v_id, config.dset, hand_score=0.1, object_score=0.01)
    
    if config.dset == 'epic':
        dset = EPICDataset(config.root)
    else:
        dset = SingleVideoDataset(config.root, config.v_id, config)
        
    config.total_frames = dset.video_length[config.v_id]
    tracker = TrackManager(dset=dset, root=config.root, config=config)
    
    for el in tqdm.tqdm(dataset, total=len(dataset)):
        left_object_bbox = bbox_xyxy_2_bbox_xywh(el['left']['bbox'])
        if left_object_bbox[2]*left_object_bbox[3] == 0:
            left_object_bboxs = []
        else:
            left_object_bboxs = left_object_bbox
            
        right_object_bbox = bbox_xyxy_2_bbox_xywh(el['right']['bbox']) 
        if right_object_bbox[2]*right_object_bbox[3] == 0:
            right_object_bboxs = []
        else:
            right_object_bboxs = right_object_bbox
        
        frame_num = el['frame_num']        
        tracker.step(el['frame'], left_object_bboxs, right_object_bboxs, el['left']['hand_presence'], el['right']['hand_presence'], el['left']['score'], el['right']['score'], frame_num)

        