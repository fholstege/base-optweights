


import os
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import h5py
import multiprocessing as mp
import tempfile


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

def save_chunk(args):
    temp_dir, dataset_name, start_idx, chunk, compression = args
    temp_filename = os.path.join(temp_dir, f"{dataset_name}_{start_idx}.h5")
    with h5py.File(temp_filename, 'w') as f:
        print("Saving data at index: {}".format(start_idx))
        f.create_dataset(dataset_name, data=chunk.numpy(), compression=compression)
    return temp_filename

def save_dataset_parallel(filename, dataset_name, dataloader, compression="gzip", compression_opts=4):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save chunks in parallel
        pool = mp.Pool(processes=mp.cpu_count())
        start_idx = 0
        args_list = []

        for batch in dataloader:
            print("Preparing data: {}/{}".format(start_idx, len(dataloader.dataset)))
            args_list.append((temp_dir, dataset_name, start_idx, batch, compression))
            start_idx += len(batch)

        temp_files = pool.map(save_chunk, args_list)
        pool.close()
        pool.join()

        # Combine temporary files
        with h5py.File(filename, 'a') as main_file:

            # first, define the chunk size - which is (number of samples, image size)
            chunk_size =(128, 3, 224, 224)
            main_file.create_dataset(dataset_name, shape=(len(dataloader.dataset),) + tuple(dataloader.dataset[0].shape), 
                                     dtype='float32', compression=compression, compression_opts=compression_opts, chunks=chunk_size)
            
            for temp_file in temp_files:
                with h5py.File(temp_file, 'r') as f:
                    data = f[dataset_name][:]
                    start = int(temp_file.split('_')[-1].split('.')[0])
                    end = start + len(data)
                    print("Combining data from {} to {}".format(start, end))
                    main_file[dataset_name][start:end] = data


class CelebA_loader(Dataset):
    def __init__(self, image_paths, transform_func):
        self.image_paths = image_paths
        self.transform_func = transform_func

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return load_image(self.image_paths[idx], self.transform_func)

def create_dataloader(image_paths, transform_func, batch_size):
    dataset = CelebA_loader(image_paths, transform_func)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4)

def save_to_hdf5(file_path, data_dict):
    with h5py.File(file_path, 'w') as f:
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                f.create_dataset(key, data=value.numpy(), compression="gzip", compression_opts=9)
            else:
                f.create_dataset(key, data=value)


def transforms_CelebA(training, target_resol, add_random_crop, add_random_flip): 
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (target_resol, target_resol)

    random_crop_func = transforms.RandomResizedCrop(target_resol)  # randomly flip image horizontally, slight
    random_flip = transforms.RandomHorizontalFlip()  # randomly flip image horizontally

    augmentations = []
    if add_random_crop and training:
        augmentations.append(random_crop_func)
    if add_random_flip and training:
        augmentations.append(random_flip)

    start_transform = [transforms.CenterCrop(orig_min_dim),  # crop the center of the image to resol x resol
                        transforms.Resize(target_resolution)]  # resize 
    
    end_transform = [transforms.ToTensor(),  # implicitly divides by 255
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]  
    
    transform =  transforms.Compose(start_transform + augmentations + end_transform) if augmentations else transforms.Compose(start_transform + end_transform)


    return transform



  



def main(main_task_name, concept_name, save_dir,   add_random_crop=True, add_random_flip=True, batch_size=512):

    # load the meta data for the spurious corr
    df_metadata_file= 'celebA/metadata_celeba_{}_{}.csv'.format(main_task_name, concept_name)
    df_metadata = pd.read_csv(df_metadata_file)
    df_metadata_train = df_metadata[df_metadata['split'] == 0]
    df_metadata_val = df_metadata[df_metadata['split'] == 1]
    df_metadata_test = df_metadata[df_metadata['split'] == 2]

    # load the y
    y_train = torch.from_numpy(df_metadata_train['y_m'].values)
    y_val = torch.from_numpy(df_metadata_val['y_m'].values)
    y_test = torch.from_numpy(df_metadata_test['y_m'].values)

    # load the c
    c_train = torch.from_numpy(df_metadata_train['y_c'].values)
    c_val = torch.from_numpy(df_metadata_val['y_c'].values)
    c_test = torch.from_numpy(df_metadata_test['y_c'].values)

    # define transform functions
    train_transform_func = transforms_CelebA(training=True,target_resol=224, add_random_crop=add_random_crop, add_random_flip=add_random_flip)
    test_transform_func = transforms_CelebA(training=False,target_resol=224, add_random_crop=add_random_crop, add_random_flip=add_random_flip)

    # Create dataloaders
    train_image_paths = ['celebA/images/{}'.format(df_metadata_train['img_filename'].iloc[i]) for i in range(len(df_metadata_train))]
    val_image_paths = ['celebA/images/{}'.format(df_metadata_val['img_filename'].iloc[i]) for i in range(len(df_metadata_val))]
    test_image_paths = ['celebA/images/{}'.format(df_metadata_test['img_filename'].iloc[i]) for i in range(len(df_metadata_test))]

    train_loader = create_dataloader(train_image_paths, train_transform_func, batch_size)
    val_loader = create_dataloader(val_image_paths, test_transform_func, batch_size)
    test_loader = create_dataloader(test_image_paths, test_transform_func, batch_size)



    # set the filename
    filename = 'data_celebA_{}_{}_random_crop_{}_random_flip_{}'.format(
        main_task_name, concept_name, add_random_crop, add_random_flip)
    
    # Save data in chunks
    filename = os.path.join(save_dir, filename) + '.h5'

    print("Saving data to: {}".format(filename))


    with h5py.File(filename, 'w') as f:
        # Save metadata
        f.create_dataset('c_train', data=c_train.numpy())
        f.create_dataset('y_train', data=y_train.numpy())
        f.create_dataset('c_val', data=c_val.numpy())
        f.create_dataset('y_val', data=y_val.numpy())
        f.create_dataset('c_test', data=c_test.numpy())
        f.create_dataset('y_test', data=y_test.numpy())

    print("Metadata saved successfully!")
    
    # Save images in parallel
    save_dataset_parallel(filename, 'X_train', train_loader)
    save_dataset_parallel(filename, 'X_val', val_loader)
    save_dataset_parallel(filename, 'X_test', test_loader)

    print("Data saved successfully!")

  




   



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset preparation')
    parser.add_argument('--save_dir', '-d', help='Where to save the new datasets')
    parser.add_argument('--main_task_name', help ='Name of the main task')
    parser.add_argument('--concept_name', help ='Name of the concept')
    parser.add_argument('--add_random_crop', type=str, default='True', help='Whether to add random crop to the images')
    parser.add_argument('--add_random_flip', type=str, default='True', help='Whether to add random flip to the images')
    args = parser.parse_args()

    def str_to_bool(text):
        if text.lower() == 'true':
            return True
        elif text.lower() == 'false':
            return False
    
    args.add_random_crop = str_to_bool(args.add_random_crop)
    args.add_random_flip = str_to_bool(args.add_random_flip)

   

    # Run the main function
    main(args.main_task_name, args.concept_name, args.save_dir, add_random_crop=args.add_random_crop, add_random_flip=args.add_random_flip, train_val_split=args.train_val_split)
