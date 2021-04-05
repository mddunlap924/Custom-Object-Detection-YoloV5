import os
import numpy as np
import pandas as pd
from ensemble_boxes import nms, soft_nms, weighted_boxes_fusion, non_maximum_weighted, nms_method
from datetime import datetime


def split_string(line, height, width, *, normalize=True):
    line_ = [line.split(' ')[i:i + 6] for i in range(0, len(line.split(' ')), 6)]
    label = []
    score = []
    box = []
    for det in line_:
        label.append(int(det[0]))
        score.append(float(det[1]))
        box_ = [int(i) for i in det[2:]]
        if normalize:
            box_[0] = box_[0] / width
            box_[1] = box_[1] / height
            box_[2] = box_[2] / width
            box_[3] = box_[3] / height
            if np.array(box_).max() > 1.0:
                print(f'Normalization Error - Check width and height of image')
        box.append(box_)

    return label, score, box


def det_array_to_string(labels, scores, boxes, height, width, det_length):
    det_string_ = []
    for i, (label, score, box) in enumerate(zip(labels, scores, boxes)):
        box[0] = box[0] * width
        box[1] = box[1] * height
        box[2] = box[2] * width
        box[3] = box[3] * height

        if len(labels) > 1 and label == 14:
            if 1 in det_length:
                score_ = str(0.5)
            else:
                score_ = str(0)
        elif len(labels) == 1 and label == 14:
            score_ = str(1)
        else:
            score_ = str(score)

        det_ = [str(int(label)), score_]
        for box_value in [str(int(ii)) for ii in box]:
            det_.append(box_value)
        det_string_.append(' '.join([i for i in det_]))

    det_string = ' '.join([i for i in det_string_])

    return det_string


""" USER INPUTS """
# Put in the path location of the submission file for as many results you want to ensemble
# Input submission paths into "files_to_analyze" as a list
files_to_analyze = [
    '/home/dunlap/kaggle/VinBigData_YOLO/results/Ensemble_Single_Models/27_03_2021 09:01:37',
    '/home/dunlap/kaggle/VinBigData_YOLO/results/Ensemble_Single_Models/27_03_2021 10:23:04',
    '/home/dunlap/kaggle/VinBigData_YOLO/results/Ensemble_Single_Models/27_03_2021 21:11:34',
    '/home/dunlap/kaggle/VinBigData_YOLO/results/Ensemble_Single_Models/23_03_2021 16:17:29',
    '/home/dunlap/kaggle/VinBigData_YOLO/results/Ensemble_Single_Models/23_03_2021 16:26:22',
    '/home/dunlap/kaggle/VinBigData_YOLO/results/Ensemble_Single_Models/25_03_2021 16:10:32',
    '/home/dunlap/kaggle/VinBigData_YOLO/results/Ensemble_Single_Models/25_03_2021 20:04:00',
]

# Path location of test_meta.csv file
test_meta_path = '/home/dunlap/kaggle/VinBigData_YOLO/Data'

# Concatenate all model results into a single list (True) or assign as separate models (False) and weights can be
# assigned
concatenate_into_single_list = False
weights = None  # Update if you want to assign different weights to each model during ensemble

# Select from different box ensemble techniques
# Possible selections: 'non_maximum_weighted' 'weighted_boxes_fusion' 'nms'
fusion_technique = 'nms_method'

# IOU Threshold for combining boxes and confidence threshold for overlooking (not using) certain boxes
iou_thr = 0.4
skip_box_thr = 0.001


""" Calculations """
# Load Dataframes
dfs = []
for i, file_ in enumerate(files_to_analyze):
    dfs.append(pd.read_csv(os.path.join(file_, 'submission.csv')))

df_sub = dfs[0].copy()
df_sub['PredictionString'] = ''

# Output Directory - If you want something else just add as user input up above
# I made time stamped folder names for easy reference (I'll likely update into better workflow in future)
outdir = os.path.join(os.getcwd(), 'results', 'Ensemble_Single_Models',
                      str(datetime.now().strftime("%d_%m_%Y %H:%M:%S")))

img_ids = dfs[0]['image_id'].tolist()
NORMAL = '14 1 0 0 1 1'
test_meta = pd.read_csv(os.path.join(test_meta_path, 'test_meta.csv'))

# Loop through each image (by img_id) and ensemble results from each submission file given in files_to_analyze
for ii, img_id in enumerate(img_ids):

    label = []
    score = []
    box = []
    det_lengths = []
    for i, df in enumerate(dfs):
        pred_ = df.loc[df['image_id'] == img_id]['PredictionString'].iloc[0]

        sub_idx = df_sub.loc[df_sub.isin([img_id]).any(axis=1)].index.tolist()[0]
        _, height, width = test_meta.loc[test_meta['image_id'] == img_id].iloc[0].values

        # Get labels for each model
        label_, score_, box_ = split_string(pred_, height, width)
        label.append(label_)
        score.append(score_)
        box.append(box_)
        det_lengths.append(len(label_))

    # Ensemble is performed differently if its treated as one list or separate models
    # https://github.com/ZFTurbo/Weighted-Boxes-Fusion
    if concatenate_into_single_list:
        box_input = [[j for i in box for j in i]]
        score_input = [[j for i in score for j in i]]
        label_input = [[j for i in label for j in i]]
    else:
        box_input = box
        score_input = score
        label_input = label

    if fusion_technique == 'nms_method':
        boxes, scores, labels = nms_method(box_input, score_input, label_input,
                                           weights=weights,
                                           iou_thr=iou_thr,
                                           thresh=skip_box_thr)
    elif fusion_technique == 'nms':
        boxes, scores, labels = nms(box_input, score_input, label_input,
                                           weights=weights,
                                           iou_thr=iou_thr)
    elif fusion_technique == 'soft_nms':
        boxes, scores, labels = soft_nms(box_input, score_input, label_input,
                                         weights=weights,
                                         iou_thr=iou_thr,
                                         thresh=skip_box_thr,
                                         sigma=0.5,
                                         )
    elif fusion_technique == 'non_maximum_weighted':
        boxes, scores, labels = non_maximum_weighted(box_input, score_input, label_input,
                                                     weights=weights,
                                                     iou_thr=iou_thr,
                                                     skip_box_thr=skip_box_thr)
    elif fusion_technique == 'weighted_boxes_fusion':
        boxes, scores, labels = weighted_boxes_fusion(box_input, score_input, label_input,
                                                      weights=weights,
                                                      iou_thr=iou_thr,
                                                      skip_box_thr=skip_box_thr)

    if df_sub.iloc[sub_idx, 0] == img_id:
        df_sub.iloc[sub_idx, 1] = det_array_to_string(labels, scores, boxes, height, width, det_lengths)
        det_counts_string = ' '.join([f'Model{jj}: {value}' for jj, value in enumerate(det_lengths)])
    print(f'Img {ii + 1}: {img_id}: Index {sub_idx}; Count {ii}; {det_counts_string}; Final {len(labels)}')


""" Create output directory and save new submission file """
os.makedirs(outdir, exist_ok=True)
df_sub.to_csv(os.path.join(outdir, 'submission_single_mix.csv'), index=False)
print(f'Box Fusion Completed: {outdir}')
