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

def time_to_frame(time_str, video_fps):
    minutes, sec = map(int, time_str.split(':'))
    return int((minutes * 60 + sec) * video_fps)

def interval_iou(intervals1, intervals2):
    union = set()
    intersection = set()

    for interval in intervals1:
        union.update(range(interval[0], interval[1] + 1))

    for interval in intervals2:
        interval_set = set(range(interval[0], interval[1] + 1))
        intersection.update(interval_set.intersection(union))
        union.update(interval_set)
                
    if not union:
        return 0

    return len(intersection) / len(union)

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

Q8 = pd.read_json('AMB/Q8.json')
dset = EPICDataset(args.root)
correctly_answered = 0
old_video = None
video_info = pd.read_csv('tools/EPIC_100_video_info.csv')

features_key = 'features'
cluster_key = 'cluster'

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

for _, question in tqdm(Q8.iterrows(), total=len(Q8)):
    video = question['video_id']
    
    if video != old_video:
        old_video = video
        fps = video_info[video_info['video_id'] == video]['fps'].values[0]
        summary = pd.read_json(os.path.join(args.AMEGO, 'LS_AMEGO', f'{video}.json'))
        summary_avg_features = torch.stack([torch.tensor(feat) for feat in summary[features_key].values], dim=0).squeeze(1)
    
    correct = question['correct']
    obj = list(question['question_image'].keys())[0]
    question_image = question['question_image'][obj]

    average_features = extract_location_features(dset, question_image, video, model_transforms)
    cos_sim = F.cosine_similarity(average_features, summary_avg_features)
    object_found = int(torch.argmax(cos_sim))
    object_found = summary.iloc[object_found][cluster_key]
    intervals = [[interval[0], interval[-1]] for interval in summary[summary[cluster_key] == object_found].num_frame]
        
    frame_based_answers = {}
    score = {}
    answers_intervals = []
    for ans_key in question.answers:
        possible_answer = []
        for ans_interval in question.answers[ans_key]:
            possible_answer.append([time_to_frame(ans_interval[0], fps), time_to_frame(ans_interval[1], fps)])
            
        answers_intervals.append(possible_answer)
        score[ans_key] = interval_iou(intervals, possible_answer)
    
    answer = max(score, key=score.get)
    
    results[question['id']] = [answer, correct]
    if int(answer) == int(correct):
        correctly_answered += 1
    
print(f"Average: {correctly_answered}/{len(Q8)}")    
os.makedirs('Results_AMEGO', exist_ok=True)
with open(f'Results_AMEGO/Q8.json', 'w') as f:
    json.dump(results, f)