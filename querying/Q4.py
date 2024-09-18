import pandas as pd
import torch
import os, json
import sys
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from tools.data import EPICDataset
from torchvision import transforms

def extract_object_features(dset, crops, video):
    all_features = []
    for crop in crops:
        frame, bbox = crop[0], crop[1]
        image = dset.load_image((video, frame))
        width, height = image.size
        bbox = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
        cropped_image = image.crop(bbox)
        cropped_image = dset.transform(cropped_image)
        with torch.no_grad():
            features = net(cropped_image.unsqueeze(0).cuda())
            all_features.append(features.detach().cpu())
            
    stacked_features = torch.stack(all_features, dim=0)
    average_features = torch.mean(stacked_features, dim=0, keepdim=True).squeeze(1)
    return average_features

def extract_location_features(dset, crops, video, transform):
    all_features = []
    for crop in crops:
        frame, bbox = crop[0], crop[1]
        image = dset.load_image((video, frame))
        width, height = image.size
        bbox = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
        cropped_image = image.crop(bbox)
        cropped_image = transform(cropped_image)
        with torch.no_grad():
            features = net_loc(cropped_image.unsqueeze(0).cuda())
            all_features.append(features.detach().cpu())
            
    stacked_features = torch.stack(all_features, dim=0)
    return stacked

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--root', type=str, required=True, help='EPIC root folder')
parser.add_argument('--AMEGO', type=str, required=True, help='AMEGO folder')

args = parser.parse_args()

Q4 = pd.read_json('AMB/Q4.json')
dset = EPICDataset(args.root)
correctly_answered = 0
old_video = None
video_info = pd.read_csv('tools/EPIC_100_video_info.csv')

features_key = 'features'
features_key_loc = 'features'
cluster_key = 'cluster'
cluster_key_loc = 'cluster'
net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
net = nn.DataParallel(net)
net.eval().cuda()

net_loc = torch.hub.load("facebookresearch/swag", model="vit_l16_in1k")
net_loc.head = nn.Identity()
resolution = 512
net_loc.eval().cuda()
model_transforms = transforms.Compose([
    transforms.Resize(
        resolution,
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.CenterCrop(resolution),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])
results = {}
    
for _, question in tqdm(Q4.iterrows(), total=len(Q4)):
    video = question['video_id']    
    if video != old_video:
        old_video = video
        fps = video_info[video_info['video_id'] == video]['fps'].values[0]
        summary = pd.read_json(os.path.join(args.AMEGO, 'HOI_AMEGO', f'{video}.json'))
        summary_loc = pd.read_json(os.path.join(args.AMEGO, 'LS_AMEGO', f'{video}.json'))
        summary_avg_features = torch.stack([torch.tensor(feat) for feat in summary[features_key].values], dim=0).squeeze(1)
    
    correct = question['correct']
    obj = list(question['question_image'].keys())[0]
    question_image = question['question_image'][obj]
    predicted_seq = {}

    average_features = extract_object_features(dset, question_image, video)
    cos_sim = F.cosine_similarity(average_features, summary_avg_features)
    object_found = int(torch.argmax(cos_sim))
    
    object_found = summary.iloc[object_found][cluster_key]
    object_apps = summary[summary[cluster_key] == object_found].sort_values(by='num_frame', key=lambda x: x.apply(lambda y: y[0]), ignore_index=True)
    if 'take' in question['question']:
        object_frame = object_apps.iloc[0]['num_frame'][0]
    else:
        object_frame = object_apps.iloc[-1]['num_frame'][-1]
    
    location = summary_loc.iloc[summary_loc['num_frame'].apply(lambda num_frames: abs(min(num_frames, key=lambda num: abs(num - object_frame)) - object_frame)).idxmin()][cluster_key_loc]
    location = summary_loc[summary_loc[cluster_key_loc]==location]
    location_avg_features = torch.stack([torch.tensor(feat) for feat in location[features_key_loc].values], dim=0).squeeze(1)

    score = {}
    for answer, answer_images in question['answers'].items():
        for i, answer_image in answer_images.items():
            average_features = extract_location_features(dset, answer_image, video, model_transforms)
            cos_sim = F.cosine_similarity(average_features, location_avg_features)
            max_cosine = max(cos_sim)     
            score[answer] = max_cosine
    
    answer_given = max(score, key=score.get)
    results[question['id']] = [answer_given, correct]
    if int(answer_given) == int(correct):
        correctly_answered += 1
                    
print(f"Average: {correctly_answered}/{len(Q4)}")  
os.makedirs('Results_AMEGO', exist_ok=True)
with open(f'Results_AMEGO/Q4.json', 'w') as f:
    json.dump(results, f)  