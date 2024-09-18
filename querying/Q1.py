import pandas as pd
import torch
import os
import sys, json
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from tools.data import EPICDataset
from omegaconf import OmegaConf
import argparse

def remove_subsequent_duplicates(lst):
    result = [lst[0]] if lst else []

    for elem in lst[1:]:
        if elem != result[-1]:
            result.append(elem)

    return result

def subsequence_distance(long_seq, short_seq):
    m, n = len(long_seq), len(short_seq)

    # Create a table to store the lengths of LCS
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if long_seq[i - 1] == short_seq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # The distance is the absolute difference between the length of short_seq and LCS length
    distance = abs(n - dp[m][n])

    return distance

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

Q1 = pd.read_json('AMB/Q1.json')
dset = EPICDataset(args.root)
correctly_answered = 0
old_video = None
features_key = 'features'
cluster_key = 'cluster'
net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
net = nn.DataParallel(net)
net.eval().cuda()
results = {}

for _, question in tqdm(Q1.iterrows(), total=len(Q1)):
    video = question['video_id']
        
    if video != old_video:
        old_video = video
        summary = pd.read_json(os.path.join(args.AMEGO, 'HOI_AMEGO', f'{video}.json'))
        summary_sequence = remove_subsequent_duplicates(list(summary.sort_values(by='num_frame', key=lambda x: x.apply(lambda y: y[0]), ignore_index=True)[cluster_key]))
        summary_avg_features = torch.stack([torch.tensor(feat) for feat in summary[features_key].values], dim=0).squeeze(1)
    
    correct = question['correct']
    predicted_seq = {}
    for answer, answer_images in question['answers'].items():
        predicted_seq[int(answer)] = []
        for i, answer_image in answer_images.items():
            average_features = extract_object_features(dset, answer_image, video)
            cos_sim = F.cosine_similarity(average_features, summary_avg_features)
            object_found = int(torch.argmax(cos_sim))
            object_found = summary.iloc[object_found][cluster_key]
            predicted_seq[int(answer)].append(object_found)
           
    distance = {answer: subsequence_distance(summary_sequence, predicted_seq[answer]) for answer in predicted_seq}
    answer_average = min(distance, key=distance.get)
    results[question['id']] = [answer_average, correct]
    
    if answer_average == correct:
        correctly_answered += 1
    
print(f"Average: {correctly_answered}/{len(Q1)}")    
os.makedirs('Results_AMEGO', exist_ok=True)
with open(f'Results_AMEGO/Q1.json', 'w') as f:
    json.dump(results, f)