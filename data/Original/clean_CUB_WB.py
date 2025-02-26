


import os
import sys
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
from collections import defaultdict
import argparse
from PIL import Image
import pandas as pd


def train_transforms_cub(resol, add_jitter, add_random_crop, add_random_flip): 

    resized_resol = int(resol * 256/224)
    random_crop_func = transforms.RandomResizedCrop(resol)  # randomly flip image horizontally, slight
    jitter_func = transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5))  # randomly change brightness and saturation
    random_flip = transforms.RandomHorizontalFlip()  # randomly flip image horizontally
    augmentations = []

    if add_jitter:
        augmentations.append(jitter_func)
    if add_random_crop:
        augmentations.append(random_crop_func)
    if add_random_flip:
        augmentations.append(random_flip)

    start_transform = [transforms.Resize((resized_resol, resized_resol))]  # resize to 256 x 256
    end_transform = [transforms.CenterCrop(resol),  # crop the center of the image to resol x resol
                    transforms.ToTensor(),  # implicitly divides by 255
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]  # normalize to mean 0, std 1
    
    transform =  transforms.Compose(start_transform + augmentations + end_transform) if augmentations else transforms.Compose(start_transform + end_transform)

    return transform



def test_transforms_cub(resol):
    resized_resol = int(resol * 256/224)
    transform = transforms.Compose([
            transforms.Resize((resized_resol, resized_resol)), # resize to 256 x 256
            transforms.CenterCrop(resol), # crop the center of the image to resol x resol - slightly different from the training transform
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]), #  normalize to mean 0, std 1
            ])

    return transform




def load_image(image_path, transform_func, attempts=5):

    # load the image
    for _ in range(attempts):
        try:
            img = Image.open(image_path).convert('RGB')
              # transform the image
            img = transform_func(img)

            return img
            
        except:
            Exception('Image not found: {}'.format(image_path))


  



def main(spurious_corr, save_dir, add_jitter=True, add_random_crop=True, add_random_flip=True):

    # load the meta data for the spurious corr
    df_metadata_file= 'WB_data/images/waterbirds_{}/metadata_waterbird_{}_{}_50.csv'.format(int(spurious_corr*100), int(spurious_corr*100), int(spurious_corr*100))
    df_metadata = pd.read_csv(df_metadata_file)
    df_metadata_train = df_metadata[df_metadata['split'] == 0]
    df_metadata_val = df_metadata[df_metadata['split'] == 1]
    df_metadata_test = df_metadata[df_metadata['split'] == 2]

    # load the y
    y_train = torch.from_numpy(df_metadata_train['y'].values)
    y_val = torch.from_numpy(df_metadata_val['y'].values)
    y_test = torch.from_numpy(df_metadata_test['y'].values)

    # load the c
    c_train = torch.from_numpy(df_metadata_train['place'].values)
    c_val = torch.from_numpy(df_metadata_val['place'].values)
    c_test = torch.from_numpy(df_metadata_test['place'].values)

    # define number of samples for train, val, test
    n_train = len(y_train)
    n_val = len(y_val)
    n_test = len(y_test)

    # create empty tensors for X
    X_train = torch.zeros(n_train, 3, 224, 224)
    X_val = torch.zeros(n_val, 3, 224, 224)
    X_test = torch.zeros(n_test, 3, 224, 224)

    # define transform functions
    train_transform_func = train_transforms_cub(224, add_jitter, add_random_crop, add_random_flip)
    test_transform_func = test_transforms_cub(224)

    # loop over all train, val and test images, and load them into the tensors
    for i in range(n_train):
        print('Loading train image {} of {}'.format(i, n_train))
        image_path = 'WB_data/images/waterbirds_{}/combined/{}'.format(int(spurious_corr*100), df_metadata_train['img_filename'].iloc[i])
        print('image_path: {}'.format(image_path))
        X_train[i, :] = load_image(image_path, train_transform_func)

    for i in range(n_val):
        print('Loading val image {} of {}'.format(i, n_val))
        image_path = 'WB_data/images/waterbirds_{}/combined/{}'.format(int(spurious_corr*100), df_metadata_val['img_filename'].iloc[i])
        X_val[i, :] = load_image(image_path, test_transform_func)

    for i in range(n_test):
        print('Loading test image {} of {}'.format(i, n_test))
        image_path = 'WB_data/images/waterbirds_{}/combined/{}'.format(int(spurious_corr*100), df_metadata_test['img_filename'].iloc[i])
        X_test[i, :] = load_image(image_path, test_transform_func)
  

    # create a dictionary to store the data
    data_dict = {'X_train': X_train, 'c_train': c_train, 'y_train': y_train,
                    'X_val': X_val, 'c_val': c_val, 'y_val': y_val,
                    'X_test': X_test, 'c_test': c_test, 'y_test': y_test}
    
    # save the data
    filename = os.path.join(save_dir, 'data_WB_{}_jitter_{}_random_crop_{}_random_flip_{}.pkl'.format(int(spurious_corr*100), add_jitter, add_random_crop, add_random_flip))
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)



   



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset preparation')
    parser.add_argument('--save_dir', '-d', help='Where to save the new datasets')
    parser.add_argument('--spurious_corr', type=float, help='The spurious correlation between the concept and the main task')
    parser.add_argument('--add_jitter', type=str, default='True', help='Whether to add jitter to the images')
    parser.add_argument('--add_random_crop', type=str, default='True', help='Whether to add random crop to the images')
    parser.add_argument('--add_random_flip', type=str, default='True', help='Whether to add random flip to the images')
    args = parser.parse_args()

    def str_to_bool(text):
        if text.lower() == 'true':
            return True
        elif text.lower() == 'false':
            return False
    
    args.add_jitter = str_to_bool(args.add_jitter)
    args.add_random_crop = str_to_bool(args.add_random_crop)
    args.add_random_flip = str_to_bool(args.add_random_flip)
    

   

    # Run the main function
    main(spurious_corr=args.spurious_corr, save_dir=args.save_dir, add_jitter=args.add_jitter, add_random_crop=args.add_random_crop, add_random_flip=args.add_random_flip)

   

