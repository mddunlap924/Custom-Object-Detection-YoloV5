# VinBigData Chest X-Ray Abnormalities Detection - Yolov5

This is code I developed for the [Kaggle VinBigData Chest X-ray Abnormalities Detection competition](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection).

The useful part of this code is that it will train Yolov5 models on multiple folds of data. This typically can take hours to days depending on several factors; therefore, once this is running you can periodically check-in on your models using Tensorboards or Weights and Biases. This is useful because you can work on other tasks while this is running. I was able to train a YoloV5 – medium P6 model, in conjunction with a 2 stage classifier, that was able to score in the top 5% of the Kaggle competition.

Please refer to https://github.com/ultralytics/yolov5 for details on the YoloV5 model used here. Also, internet searches will return numerous examples and applications on this topic.

The following steps can be used to help setup a Yolov5 Object Detection model using the code given in this repository. 
  
    1. Create Data Folds for CV
        ◦ First put all images into a training and testing folder. The testing folder is for the images that are actually to be tested. For example, in this Kaggle competition it was the test images.
        ◦ It’s assumed that the images in the train and test folders all have the same height and widths (i.e., the following routines are setup to run square shaped images).
        ◦ Run “data_prep_create_folds.py” – this will create the folds of data by taking all the images from the train folder and separating those into the different folds. Each fold will have its own folder and inside that folder will be a train and validation (val) folder. These folder will be called by the yolo model during training.
          
    2. Setup Yolov5 Models
        ◦ Navigate to ./yolov5 folder
        ◦ In folder “chest_yaml” modify the chest_1024_*.yaml files for your dataset which will consist of specifying the path to the train, val folders (see step 1), number of classes, and class names.
            ▪ Create a chest_*.yaml file for each fold from step 1
        ◦ Data augmentations are controlled with the hyp.scratch.yaml file under the “data” folder
            ▪ Spend time adjusting the parameters within this file because they will affect your results and you could spend sometime here
        ◦ Run “yolov5_execute_shell.py” to begin training models
            ▪ Update directory paths as needed to run your own applications
          
    3. Ensemble Models
        ◦ Run “ensemble_models.py” to combine boxes from different object detection models or folds.
        ◦ You can further combine/ensemble with a 2 stage-classifier if you’d prefer. This is useful to help identify images that may not contain any of the classes the YoloV5 models were trained on.
