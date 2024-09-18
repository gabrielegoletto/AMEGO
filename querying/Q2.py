import pandas as pd
import torch
import os, json
import sys
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from tools.data import EPICDataset

def time_to_frame(time_str, video_fps):
    minutes, sec = map(int, time_str.split(':'))
    return int((minutes * 60 + sec) * video_fps)

def find_closest(lst, x):
    results = [el-x for el in lst]
    results = [float('inf') if el < 0 else el for el in results]
    if float('inf') in results:
        return float('inf')
    return min(results)

def contains_side(lst, side):
    for item in lst:
        if isinstance(item, list):
            if side in item:
                return True
        elif item == side:
            return True
    return False

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

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--root', type=str, required=True, help='EPIC root folder')
parser.add_argument('--AMEGO', type=str, required=True, help='AMEGO folder')

args = parser.parse_args()

Q2 = pd.read_json('AMB/Q2.json')
dset = EPICDataset(args.root)
correctly_answered = 0
old_video = None
video_info = pd.read_csv('tools/EPIC_100_video_info.csv')

features_key = 'features'
cluster_key = 'cluster'
net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
net = nn.DataParallel(net)
net.eval().cuda()
results = {}

for _, question in tqdm(Q2.iterrows(), total=len(Q2)):
    video = question['video_id']    
    if video != old_video:
        old_video = video
        fps = video_info[video_info['video_id'] == video]['fps'].values[0]
        summary = pd.read_json(os.path.join(args.AMEGO, 'HOI_AMEGO', f'{video}.json'))
    
    time = time_to_frame(question.question.split('time')[1].split('?')[0].strip(), fps)
    hand = question.question.split('hand')[0].strip().split(' ')[-1].strip()
    closest_frames = summary['num_frame'].apply(find_closest, args=(time,))
    side_contained = summary['side'].apply(contains_side, args=(hand,))
    summary['closest_frame'] = closest_frames
    summary['side_contained'] = side_contained
    side_summary = summary[summary.side_contained]
    side_summary = side_summary.sort_values('closest_frame')
    summary_avg_features = torch.stack([torch.tensor(feat) for feat in side_summary[features_key].values], dim=0).squeeze(1)
    correct = question['correct']
    obj = list(question['question_image'].keys())[0]
    question_image = question['question_image'][obj]
    
    average_features = extract_object_features(dset, question_image, video)
    cos_sim = F.cosine_similarity(average_features, summary_avg_features)
    index = int((cos_sim < 0.6).nonzero(as_tuple=False)[0])    
    next_element = side_summary.iloc[index + 1]
    next_element_feat = torch.stack([torch.tensor(feat) for feat in next_element[features_key]], dim=0).squeeze(1)
      
    score = {}  
    for ans in question.answers:
        answer = question.answers[ans]
        obj = list(answer.keys())[0]
        answer = answer[obj]
        average_features = extract_object_features(dset, answer, video)        
        cos_sim = F.cosine_similarity(average_features, next_element_feat)
        score[ans] = float(cos_sim)

    answer_given = max(score, key=score.get)
    results[question['id']] = [answer_given, correct]
    if int(answer_given) == int(correct):
        correctly_answered += 1
        
    
print(f"Average: {correctly_answered}/{len(Q2)}")    
os.makedirs('Results_AMEGO', exist_ok=True)
with open(f'Results_AMEGO/Q2.json', 'w') as f:
    json.dump(results, f)