import pandas as pd
import json
import os
from PIL import Image
from tools.transforms import default_transform
from epic_kitchens.hoa import load_detections
from epic_kitchens.hoa.types import HandSide
import torch
import numpy as np
import math
from torchvision.transforms.functional import crop

class EPICDataset():
    def __init__(self, root):
        self.root = root
        self.video_info = pd.read_csv('tools/EPIC_100_video_info.csv')
        self.video_info.set_index('video_id', inplace=True)
        self.video_fps = self.video_info['fps'].to_dict()
        self.video_length = self.video_info['num_frames'].to_dict()
        self.transform = default_transform('val')
        self.frame_shape = (456, 256)
        self.name = 'epic'
        
    def frame_path(self, img):
        v_id, f_id = img
        p_id = v_id.split('_')[0]
        file = os.path.join(self.root, f'EPIC-KITCHENS/{p_id}/rgb_frames/{v_id}/frame_{f_id:010d}.jpg') # devfair
        return file
    
    def flowformer_path(self, img):
        v_id, f_id = img
        p_id = v_id.split('_')[0]
        file = os.path.join(self.root, f'EPIC-KITCHENS/{p_id}/flowformer/{v_id}/flow_{f_id:010d}.pth') # devfair
        return file        

    def load_image(self, img):
        file = self.frame_path(img)
        img = Image.open(file).convert('RGB')
        return img
    
    def frames_root(self, v_id):
        p_id = v_id.split('_')[0]
        return os.path.join(self.root, f'EPIC-KITCHENS/{p_id}/rgb_frames/{v_id}')
    
    def flowformer_root(self, v_id):
        p_id = v_id.split('_')[0]
        return os.path.join(self.root, f'EPIC-KITCHENS/{p_id}/flowformer/{v_id}')
        
    def detections_path(self, v_id):
        p_id = v_id.split('_')[0]
        return os.path.join(self.root, f'EPIC-KITCHENS/{p_id}/hand-objects/{v_id}.pkl')


class SingleVideoDataset:
    def __init__(self, root, v_id, video_cfg):
        self.root = root
        self.v_id = v_id
        self.frame_shape = {v_id: video_cfg.frame_shape} 
        self.transform = default_transform('val')
        self.video_fps = {v_id: video_cfg.fps}
        self.video_length = {v_id: video_cfg.num_frames} 
        self.transform = default_transform('val')
        self.name = 'video'
        
    def frame_path(self, f_id):
        # Path for a specific frame in the video
        file = os.path.join(self.root, f'{self.v_id}/rgb_frames/frame_{f_id:010d}.jpg')
        return file
    
    def flowformer_path(self, f_id):
        # Path for a specific flowformer file in the video
        file = os.path.join(self.root, f'{self.v_id}/flowformer/flow_{f_id:010d}.pth')
        return file        

    def load_image(self, f_id):
        # Load and return the image for a specific frame
        file = self.frame_path(f_id)
        img = Image.open(file).convert('RGB')
        return img
    
    def frames_root(self):
        # Root path for all RGB frames in the video
        return os.path.join(self.root, f'{self.v_id}/rgb_frames')
    
    def flowformer_root(self):
        # Root path for all flowformer files in the video
        return os.path.join(self.root, f'{self.v_id}/flowformer')
        
    def detections_path(self):
        # Path for detections file, if applicable
        return os.path.join(self.root, f'{self.v_id}/hand-objects/{self.v_id}.pkl')

    
# subsampled frames desired fps for graph construction
class FrameDsetSubsampled:

    def __init__(self, root, fps, v_id, origin, **kwargs):
        
        self.v_id = v_id

        if origin == 'epic':
            self.dset = EPICDataset(root)
        else:
            self.dset = SingleVideoDataset(root, v_id, kwargs.get('video_cfg'))
                    
        frames = []
        video_fps = self.dset.video_fps[self.v_id]
        if fps > video_fps:
            desired_fps = video_fps
        elif fps <= 0:
            desired_fps = 1    
        else:
            desired_fps = fps  
        self.subsample = int(max(1, video_fps // desired_fps))
        self.vid_length = self.dset.video_length[self.v_id]
        frames += [(self.v_id, f_id) for f_id in range(1, self.vid_length+1, self.subsample)]
        self.frames = frames
        self.transform = kwargs.get('transform', True)
        
        print (f'RGB Dataset: {len(self.frames)} frames for video {self.v_id}')

    def __getitem__(self, index):
        """Get the frame

        Args:
            index (int): index of the frame in the dataset
        """
        frame = self.frames[index]
        frame_number = frame[1]
        frame = self.dset.load_image((self.v_id, frame_number))
        if self.transform:
            frame = self.dset.transform(frame)        
        return frame

    def __len__(self):
        return len(self.frames)
    
# dataset for flow data
class FlowDataset:

    def __init__(self, root, fps, v_id, origin, **kwargs):
        self.v_id = v_id

        if origin == 'epic':
            self.dset = EPICDataset(root)
        else:
            self.dset = SingleVideoDataset(root, v_id, kwargs.get('video_cfg'))
                                
        frames = []
        video_fps = self.dset.video_fps[self.v_id]
        if fps > video_fps:
            desired_fps = video_fps
        elif fps <= 0:
            desired_fps = 1      
        else:
            desired_fps = fps
        self.subsample = max(1, video_fps // desired_fps)
        self.vid_length = self.dset.video_length[self.v_id]
        frames += [(self.v_id, f_id) for f_id in range(1, self.vid_length+1, self.subsample)]
        self.frames = frames
        
        print (f'FlowDataset: {len(self.frames)} frames for video {self.v_id}')

    def __getitem__(self, index):
        """Get the frame

        Args:
            index (int): index of the frame in the dataset
        """
        frame = self.frames[index]
        frame_number = frame[1]
        img = (self.v_id, frame_number)
        flow_path = self.dset.flowformer_path(img)
        flow_tensor = torch.load(flow_path)
        return flow_tensor

    def __len__(self):
        return len(self.frames) - 1

# dataset for flow extraction
class FlowFormerDataset:

    def __init__(self, root, fps, v_id, origin, **kwargs):
        self.v_id = v_id

        if origin == 'epic':
            self.dset = EPICDataset(root)
        else:
            self.dset = SingleVideoDataset(root, v_id, kwargs.get('video_cfg'))
                    
        frames = []
        video_fps = self.dset.video_fps[self.v_id]
        if fps > video_fps:
            desired_fps = video_fps
        elif fps <= 0:
            desired_fps = 1      
        self.subsample = max(1, video_fps // desired_fps)
        self.vid_length = self.dset.video_length[self.v_id]
        frames += [(self.v_id, f_id) for f_id in range(1, self.vid_length+1, self.subsample)]
        self.frames = frames
        
        print (f'PairedFrames Dataset: {len(self.frames)} frames for video {self.v_id}')

    def __getitem__(self, index):
        """Get the frame

        Args:
            index (int): index of the frame in the dataset
        """
        frame = self.frames[index]
        frame_number = frame[1]
        frame2 = self.frames[index + 1]
        frame_number2 = frame2[1]
        
        img1 = Image.open(self.dset.frame_path((self.v_id, frame_number)))
        img2 = Image.open(self.dset.frame_path((self.v_id, frame_number2)))
        img1 = np.array(img1).astype(np.uint8)[..., :3]
        img2 = np.array(img2).astype(np.uint8)[..., :3]
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
       
        return img1, img2

    def __len__(self):
        return len(self.frames) - 1

# subsampled frames desired fps for graph construction
class ObjectFrameDsetSubsampled:

    def __init__(self, root, fps, v_id, origin, hand_score=0, object_score=0, **kwargs):
        
        self.v_id = v_id
        self.hand_score = hand_score
        self.object_score = object_score

        if origin == 'epic':
            self.dset = EPICDataset(root)
        else:
            self.dset = SingleVideoDataset(root, v_id, kwargs.get('video_cfg'))
                    
        frames = []
        video_fps = self.dset.video_fps[self.v_id]
        if fps > video_fps:
            desired_fps = video_fps
        elif fps <= 0:
            desired_fps = 1      
        self.subsample = max(1, video_fps // desired_fps)
        self.vid_length = self.dset.video_length[self.v_id]
        frames += [(self.v_id, f_id) for f_id in range(1, self.vid_length+1, self.subsample)]
        self.frames = frames
        self.hand = {'left': HandSide.LEFT, 'right': HandSide.RIGHT}
        self.detections = load_detections(self.dset.detections_path(self.v_id))
        
        for i in range(len(self.detections)):
            #temporary fix
            frame = self.dset.load_image((self.v_id, 1))
            width, height = frame.size
            self.frame_shape = (width, height)
            self.detections[i].scale(width_factor=width, height_factor=height)

        print (f'HOI Dataset: {self.vid_length} frames for video {self.v_id}')

    def __getitem__(self, index):
        """Get object interacting with the hand chosen, 
           if more hands of the same side are detected then 
           just the one with highest confidence is selected

        Args:
            index (int): index of the frame in the dataset
        """
        frame = self.frames[index]
        frame_number = frame[1]
        frame = self.dset.load_image((self.v_id, frame_number))
        object_interacting_coords = ((0, 0), (0, 0))
        score = 0
        width, height = frame.size
        # here there is -1 as frame_numbers start from 1 while the indices in the detections starts from 0
        det = self.detections[frame_number-1]
        object_presence = len([det_o for det_o in det.objects if det_o.score > self.object_score]) > 0
        results = {'left': {'object_absence': True,
                            'hand_presence': len([det_h for det_h in det.hands if det_h.side.value == self.hand['left'].value 
                                                  and det_h.score > self.hand_score]) > 0
                            }, 
                   'right': {'object_absence': True,
                             'hand_presence': len([det_h for det_h in det.hands if det_h.side.value == self.hand['right'].value
                                                    and det_h.score > self.hand_score]) > 0
                             },
                   'frame': frame,
                   'frame_num': frame_number,
                   'frame_shape': (width, height),
                   }
        
        for side in ['left', 'right']:
            if not results[side]['hand_presence'] or not object_presence:
                results[side]['score'] = 0
                results[side]['bbox'] = ((0, 0), (0, 0))
                results[side]['frame_object'] = self.dset.transform(frame)
            else:
                interacting = det.get_hand_object_interactions(self.object_score, self.hand_score, True)
                interacting = [obj_idx for hand_idx, obj_idx in interacting.items() if det.hands[hand_idx].side.value == self.hand[side].value]
                assert len(interacting) <= 1, "Error in detecting interacting object"
                if len(interacting) == 0:
                    results[side]['score'] = 0
                    results[side]['bbox'] = ((0, 0), (0, 0))
                    results[side]['frame_object'] = self.dset.transform(frame)
                else:
                    results[side]['object_absence'] = False
                    assert len(interacting) == 1; "More than 1 object interacting with the desired hand side"
                    object_interacting_coords = det.objects[interacting[0]].bbox.coords_int
                    results[side]['bbox'] = object_interacting_coords
                    score = det.objects[interacting[0]].score
                    results[side]['score'] = score
                    frame_cropped = frame.crop((object_interacting_coords[0][0], object_interacting_coords[0][1], object_interacting_coords[1][0], object_interacting_coords[1][1]))
                    if frame_cropped.size[0] == 0 or frame_cropped.size[1] == 0:
                        frame_cropped = frame
                        results[side]['object_absence'] = True

                    results[side]['frame_object'] = self.dset.transform(frame_cropped)
        
        return results
        
    def __len__(self):
        return len(self.frames)
    