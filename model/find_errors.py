import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import image_utils
from keras.applications.vgg16 import preprocess_input
from urllib.parse import unquote

import csv
from PIL import UnidentifiedImageError

# def get_error_files(dataset,set_type):
#     X_train_data = []
#     error_files = []
#     count = 1
#     for fp in dataset['file_path']:
#         print(count)
#         try:
#             im = image_utils.load_img(fp, target_size=(224, 224))
#             im_array = image_utils.img_to_array(im)
#             X_train_data.append(im_array)
#         except Exception as e:
#             print(f"Error loading image at file path {fp}, exception: {e}")
#             error_files.append(fp)
#             continue
#         count += 1
#
#     print(f"Error Files: {len(error_files)} \n")
#     with open(f'error_files_{set_type}.csv', 'w', newline='') as file:
#         writer = csv.writer(file)
#         for error in error_files:
#             print(error)
#             print("\n")
#             writer.writerow(error)
#
#     return error_files
def get_error_files(dataset,set_type, batch_size=32):
    num_images = len(dataset)
    X_train_data = []
    error_files = []
    count = 1
    batch_count = 1
    for i in range(1,num_images,batch_size):
        batch_paths = dataset['file_path'].iloc[i:i + batch_size]
        batch_images = []
        for fp in batch_paths:
            print(count)
            try:
                im = image_utils.load_img(fp, target_size=(224, 224))
                im_array = image_utils.img_to_array(im)
                batch_images.append(im_array)
            except Exception as e:
                print(f"Error loading image at file path {fp}, exception: {e}")
                error_files.append(fp)
                continue
            count += 1
        batch_images = np.array(batch_images)
        print(f"Batch number {batch_count} created")
        batch_count+=1
        X_train_data.append(batch_images)

    print(f"Error Files: {len(error_files)} \n")
    with open(f'error_files_{set_type}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for error in error_files:
            print(error)
            print("\n")
            writer.writerow(error)
    X_train_data = np.concatenate(X_train_data, axis=0)
    X_train_data = preprocess_input(X_train_data)

    return error_files
print("Setting model file paths")
model_path = os.path.join(os.getcwd(), 'complete', 'multi-classifier-2')
data_path = os.path.join(model_path,'data')
dataset_dir = "D:\\Programming\\Spider-Prediction\\dataset\\Updated Dataset\\Spider Identification Images"

print("Setting data file paths")
train_df_file = os.path.join(data_path,'train_df_errorfinding.csv')
val_df_file = os.path.join(data_path,'val_df.csv')
test_df_file = os.path.join(data_path,'test_df.csv')

print("Loading data files")
train_df = pd.read_csv(train_df_file)
val_df = pd.read_csv(val_df_file)
test_df = pd.read_csv(test_df_file)

print("Retrieving file path columns")
X_train = train_df.loc[:,['file_path']]
X_val = val_df.loc[:,['file_path']]
X_test = test_df.loc[:,['file_path']]

# print("***********************RUNNING ON TRAIN SET***********************")
# X_train_errors = get_error_files(X_train,'train')
print("***********************RUNNING ON VAL SET***********************")
X_val_errors = get_error_files(X_val,'val')
print("***********************RUNNING ON TEST SET***********************")
X_test_errors = get_error_files(X_test,'test')