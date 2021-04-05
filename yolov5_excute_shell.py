import os
import yolo_pred_to_sub

# If you want to track your models on Weights and Biases input your api key
# https://wandb.ai/site
# os.environ["WANDB_API_KEY"] = ''


# Observe results on Tensorboards using the below terminal command:
# tensorboard --logdir runs/train


""" Various Functions/Methods """


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def remove_images(path):
    files_in_directory = os.listdir(path)

    filtered_files = [file for file in files_in_directory if file.endswith(".png")]

    for file in filtered_files:
        path_to_file = os.path.join(path, file)
        os.remove(path_to_file)

    return


""" USER INPUTS """
folds = [0, 1, 2, 3, 4]  # Add/remove folds you want to train on
data_folder = '/home/dunlap/kaggle/VinBigData_YOLO/chest_yolo'  # folders containing folds of data imgs
img_size = 1024  # img resolution used
yolo_model = 'm6'  # yolo model (https://github.com/ultralytics/yolov5/tree/master/models)
batch_size = 16  # adjust as needed


""" Calculations """
for fold in folds:

    # Train Model
    command_train = f'python ./yolov5/train.py --epochs 36 ' \
              f'--batch-size {batch_size} ' \
              f'--cfg ./yolov5/chest_yaml/yolov5{yolo_model}.yaml ' \
              f'--data ./yolov5/chest_yaml/chest_{img_size}_{fold}.yaml ' \
              f'--weights ./yolov5/chest_yaml/yolov5{yolo_model}.pt ' \
              f'--img {img_size} ' \
              f'--cache '
    os.system(command_train)

    # Test Inference
    inf_folder = newest('/home/dunlap/kaggle/VinBigData_YOLO/runs/train')
    img_size_name = str(img_size) + 'x' + str(img_size)
    command_test = f'python ./yolov5/detect.py ' \
                   f'--weights ./runs/train/{inf_folder.split("/")[-1]}/weights/last.pt ' \
                   f'--img-size {int(img_size * 1.30)} ' \
                   f'--conf-thres 0.0001 ' \
                   f'--source ./Data/vinbigdata-chest-xray-resized-png-{img_size_name}/test/ ' \
                   f'--iou-thres 0.45 ' \
                   f'--save-txt ' \
                   f'--save-conf ' \
                   f'--augment'
    os.system(command_test)

    # Convert Test Inference Text Labels into Submission File
    inf_path = f'/home/dunlap/kaggle/VinBigData_YOLO/runs/detect/{inf_folder.split("/")[-1]}'
    yolo_pred_to_sub1.text_labels_to_sub(inf_folder.split("/")[-1], img_size)

    # Delete Images with BBox to keep file space down
    remove_images(inf_path)

    print(f'Finished Fold {fold}')

print('check point')
