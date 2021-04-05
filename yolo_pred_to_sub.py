import os
import numpy as np
import pandas as pd

def text_labels_to_sub(folder_name, IMG_SIZE):
    DATA_TO_CONVERT = f'/home/dunlap/kaggle/VinBigData_YOLO/runs/detect/{folder_name}/labels'

    test_meta = pd.read_csv('/Data/test_meta.csv')
    test_meta.columns = ['image_id', 'h', 'w']

    df_sub = pd.read_csv('/Data/sample_submission.csv')
    df_sub['PredictionString'] = ''

    def resize_box(box, h, w):

        x_min = (box[0] - (box[2] / 2)) * w
        y_min = (box[1] - (box[3] / 2)) * h
        x_max = (box[0] + (box[2] / 2)) * w
        y_max = (box[1] + (box[3] / 2)) * h

        bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]

        if bbox[2] > w:
            print('width size error')
        if bbox[3] > h:
            print('height size error')

        return bbox

    def det_to_string(scores, labels, boxes):

        det_string_ = []
        for _, (label, score, box) in enumerate(zip(labels, scores, boxes)):
            det_ = [str(label), str(score)]
            for box_value in box:
                det_.append(str(box_value))
            det_string_.append(' '.join([i for i in det_]))

        det_string = ' '.join([i for i in det_string_])

        return det_string


    img_ids = df_sub.image_id.tolist()
    results = []
    header_names = ['class_id', 'xc', 'yc', 'width', 'height', 'score']
    for i, img_id in enumerate(img_ids):
        result_ = []
        data = pd.read_table(os.path.join(DATA_TO_CONVERT, img_id + '.txt'), delimiter=' ', header=None,
                             names=header_names)

        sub_idx = df_sub.loc[df_sub.isin([img_id]).any(axis=1)].index.tolist()[0]

        _, height, width = test_meta.loc[test_meta['image_id'] == img_id].iloc[0].values

        class_ = []
        box_ = []
        score_ = []
        for idx, row in data.iterrows():
            class_.append(int(row.class_id))
            box_.append(resize_box([row.xc, row.yc, row.width, row.height], height, width))
            score_.append(row.score)

        det_string = det_to_string(score_, class_, box_)

        if df_sub.iloc[sub_idx, 0] == img_id:
            df_sub.iloc[sub_idx, 1] = det_string
        print(f'{i}; {img_id}: Num Preds {len(class_)}')

    df_sub.to_csv(os.path.join(DATA_TO_CONVERT, 'submission.csv'), index=False)
    print('Completed Making Submission File')


