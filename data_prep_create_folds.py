import pandas as pd
import numpy as np
import logging
from tqdm.notebook import tqdm
import os
from glob import glob
import shutil as sh
from ml_stratifiers import MultilabelStratifiedKFold

""" Notes """
"""This is the first file to execute in a project because it will setup the K-Fold files for training models. This 
will create a directory for each fold and within each folder is a train and val folder. The train and val images are 
placed within each folder. Yes, this is not hard disk efficient and other more optimal routines could be developed 
but if you have the space and not too many images then this can only be performed once and then move onto other 
tasks. """
# This py file was modified from the following Kaggle VinBigData public Notebooks:
# https://www.kaggle.com/nxhong93/yolov5-chest-512
# https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/207955


""" Various Methods """


# Resize labels
def label_resize(org_size, img_size, *bbox):
    x0, y0, x1, y1 = bbox
    x0_new = int(np.round(x0 * img_size[1] / org_size[1]))
    y0_new = int(np.round(y0 * img_size[0] / org_size[0]))
    x1_new = int(np.round(x1 * img_size[1] / org_size[1]))
    y1_new = int(np.round(y1 * img_size[0] / org_size[0]))
    return x0_new, y0_new, x1_new, y1_new


# Split a dataframe using a multi-label stratified k-fold technique
# https://github.com/trent-b/iterative-stratification
def split_df(df):
    kf = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    df['id'] = df.index
    annot_pivot = pd.pivot_table(df, index=['image_id'], columns=['class_id'],
                                 values='id', fill_value=0, aggfunc='count') \
        .reset_index().rename_axis(None, axis=1)
    for fold, (train_idx, val_idx) in enumerate(kf.split(annot_pivot,
                                                         annot_pivot.iloc[:, 1:(1 + df['class_id'].nunique())])):
        annot_pivot[f'fold_{fold}'] = 0
        annot_pivot.loc[val_idx, f'fold_{fold}'] = 1
    return annot_pivot


def create_file(df, df_split, train_file, train_folder, fold):
    os.makedirs(f'{train_file}/Fold{fold}/labels/train/', exist_ok=True)
    os.makedirs(f'{train_file}/Fold{fold}/images/train/', exist_ok=True)
    os.makedirs(f'{train_file}/Fold{fold}/labels/val/', exist_ok=True)
    os.makedirs(f'{train_file}/Fold{fold}/images/val/', exist_ok=True)

    list_image_train = df_split[df_split[f'fold_{fold}'] == 0]['image_id']
    train_df = df[df['image_id'].isin(list_image_train)].reset_index(drop=True)
    val_df = df[~df['image_id'].isin(list_image_train)].reset_index(drop=True)

    for train_img in tqdm(train_df.image_id.unique()):
        with open(f'{train_file}/Fold{fold}/labels/train/{train_img}.txt', 'w+') as f:
            row = train_df[train_df['image_id'] == train_img] \
                [['class_id', 'x_center', 'y_center', 'width', 'height']].values
            row[:, 1:] /= SIZE
            row = row.astype('str')
            for box in range(len(row)):
                text = ' '.join(row[box])
                f.write(text)
                f.write('\n')
        sh.copy(f'{train_folder}/{train_img}.png',
                f'{train_file}/Fold{fold}/images/train/{train_img}.png')

    for val_img in tqdm(val_df.image_id.unique()):
        with open(f'{train_file}/Fold{fold}/labels/val/{val_img}.txt', 'w+') as f:
            row = val_df[val_df['image_id'] == val_img] \
                [['class_id', 'x_center', 'y_center', 'width', 'height']].values
            row[:, 1:] /= SIZE
            row = row.astype('str')
            for box in range(len(row)):
                text = ' '.join(row[box])
                f.write(text)
                f.write('\n')
        sh.copy(f'{train_folder}/{val_img}.png',
                f'{train_file}/Fold{fold}/images/val/{val_img}.png')


""" USER INPUTS """
IMGS_TO_USE = '1024'  # Different images were tested based on their resolution (height x width)
TRAIN_PATH_CSV = '/home/dunlap/kaggle/VinBigData_YOLO/Data/train.csv'
SUB_PATH = '/home/dunlap/kaggle/VinBigData_YOLO/Data/sample_submission.csv'
# Keep the images to be split into folds (i.e., all original images in a folder called "Data")
ORIGINAL_IMGS_PATH = '/home/dunlap/kaggle/VinBigData_YOLO/Data/'
FOLDS = 5
SEED = 42

""" Calculations """
img_size_str = str(IMGS_TO_USE) + 'x' + str(IMGS_TO_USE)
# Path to imgs (in this is useful if you want to use different resolution size images)
IMGS_FOLDER = f'vinbigdata-chest-xray-resized-png-{img_size_str}'
DATA_PATH = os.path.join(ORIGINAL_IMGS_PATH, IMGS_FOLDER)
RESIZE_PATH = DATA_PATH
MAIN_PATH = os.getcwd()
TRAIN_DICOM_PATH = os.path.join(MAIN_PATH, 'train')
TEST_DICOM_PATH = os.path.join(MAIN_PATH, 'test')
TRAIN_PATH = os.path.join(RESIZE_PATH, 'train')
TEST_PATH = os.path.join(RESIZE_PATH, 'test')
TRAIN_META_PATH = os.path.join(DATA_PATH, 'train_meta.csv')
TEST_META_PATH = os.path.join(DATA_PATH, 'test_meta.csv')
SIZE = int(IMGS_TO_USE)
IMG_SIZE = (SIZE, SIZE)
ACCULATION = 1
MOSAIC_RATIO = 0.4

# Setup a logger
logging.basicConfig(format='%(asctime)s +++ %(message)s',
                    datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()

""" Read File """
train_list = glob(f'{TRAIN_PATH}/*.png')
test_list = glob(f'{TEST_PATH}/*.png')
logger.info(f'Train have {len(train_list)} file and test have {len(test_list)}')
size_df = pd.read_csv(TRAIN_META_PATH)
size_df.columns = ['image_id', 'h', 'w']
train_df = pd.read_csv(TRAIN_PATH_CSV)
train_df = train_df.merge(size_df, on='image_id', how='left')
train_df[['x_min', 'y_min']] = train_df[['x_min', 'y_min']].fillna(0)
train_df[['x_max', 'y_max']] = train_df[['x_max', 'y_max']].fillna(1)
train_df.tail()

# Create a normal image training df
train_normal = train_df[train_df['class_name'] == 'No finding'].reset_index(drop=True)
train_normal['x_min_resize'] = 0
train_normal['y_min_resize'] = 0
train_normal['x_max_resize'] = 1
train_normal['y_max_resize'] = 1
train_abnormal = train_df[train_df['class_name'] != 'No finding'].reset_index(drop=True)
train_abnormal[['x_min_resize', 'y_min_resize', 'x_max_resize', 'y_max_resize']] = train_abnormal \
    .apply(lambda x: label_resize(x[['h', 'w']].values, IMG_SIZE, *x[['x_min', 'y_min', 'x_max', 'y_max']].values),
           axis=1, result_type="expand")
train_abnormal['x_center'] = 0.5 * (train_abnormal['x_min_resize'] + train_abnormal['x_max_resize'])
train_abnormal['y_center'] = 0.5 * (train_abnormal['y_min_resize'] + train_abnormal['y_max_resize'])
train_abnormal['width'] = train_abnormal['x_max_resize'] - train_abnormal['x_min_resize']
train_abnormal['height'] = train_abnormal['y_max_resize'] - train_abnormal['y_min_resize']
train_abnormal['area'] = train_abnormal.apply(
    lambda x: (x['x_max_resize'] - x['x_min_resize']) * (x['y_max_resize'] - x['y_min_resize']), axis=1)


""" K-Fold """
size_df = pd.read_csv(TRAIN_META_PATH)
size_df.columns = ['image_id', 'h', 'w']

fold_csv = split_df(train_abnormal)
fold_csv = fold_csv.merge(size_df, on='image_id', how='left')
fold_csv.head(10)


""" Create Folds """
# Create the different directories for each fold
# Structure: Fold 0 - train, val; and each folder are the images
# I manually update the last input on the below line and run each fold
# Folds are stored in folder names "chest_yolo" (updated as desired)
create_file(train_abnormal, fold_csv, './chest_yolo', TRAIN_PATH, 0)
print('Finished processing data')
