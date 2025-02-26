


import os
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import multiprocessing as mp
import tempfile

def transforms_chexpert(resol): 


    transform = transforms.Compose([
            transforms.Resize(resol),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

def save_chunk(args):
    temp_dir, dataset_name, start_idx, chunk, compression = args
    temp_filename = os.path.join(temp_dir, f"{dataset_name}_{start_idx}.h5")
    with h5py.File(temp_filename, 'w') as f:
        print("Saving data at index: {}".format(start_idx))
        f.create_dataset(dataset_name, data=chunk.numpy(), compression=compression)
    return temp_filename


def save_dataset_parallel(filename, dataset_name, dataloader, chunk_dim, compression="gzip", compression_opts=4, processes=2, split_in_groups_of=100):
    with tempfile.TemporaryDirectory() as temp_dir:
        pool = mp.Pool(processes=processes)
        start_idx = 0
        args_list = []
        batch_counter = 0

        # Create main file and dataset once
        with h5py.File(filename, 'a') as main_file:
            chunk_size = (chunk_dim, 3, 224, 224)
            if dataset_name not in main_file:
                main_file.create_dataset(dataset_name, 
                                      shape=(len(dataloader.dataset),) + tuple(dataloader.dataset[0].shape),
                                      dtype='float32', 
                                      compression=compression, 
                                      compression_opts=compression_opts,
                                      chunks=chunk_size)

        # Process batches
        for batch in dataloader:
            print(f"Preparing data: {start_idx}/{len(dataloader.dataset)}")
            args_list.append((temp_dir, dataset_name, start_idx, batch, compression))
            start_idx += len(batch)
            batch_counter += 1

            # Save after every split_in_groups_of batches
            if batch_counter == split_in_groups_of:
                temp_files = pool.map(save_chunk, args_list)
                
                # Save to main file
                with h5py.File(filename, 'a') as main_file:
                    for temp_file in temp_files:
                        with h5py.File(temp_file, 'r') as f:
                            data = f[dataset_name][:]
                            start = int(temp_file.split('_')[-1].split('.')[0])
                            end = start + len(data)
                            print(f"Combining data from {start} to {end}")
                            main_file[dataset_name][start:end] = data
                        os.remove(temp_file)
                
                # Reset for next group
                args_list = []
                batch_counter = 0
                print('Done with a group')
            
            print('Batch counter: {}'.format(batch_counter))

        # Process remaining batches
        if args_list:
            temp_files = pool.map(save_chunk, args_list)
            with h5py.File(filename, 'a') as main_file:
                for temp_file in temp_files:
                    with h5py.File(temp_file, 'r') as f:
                        data = f[dataset_name][:]
                        start = int(temp_file.split('_')[-1].split('.')[0])
                        end = start + len(data)
                        print(f"Combining data from {start} to {end}")
                        main_file[dataset_name][start:end] = data
                    os.remove(temp_file)

        pool.close()
        pool.join()



class chexpert_loader(Dataset):
    def __init__(self, image_paths, transform_func):
        self.image_paths = image_paths
        self.transform_func = transform_func

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return load_image(self.image_paths[idx], self.transform_func)

def create_dataloader(image_paths, transform_func, batch_size):
    dataset = chexpert_loader(image_paths, transform_func)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4)

    
def main(save_dir='../Cleaned/chexpert', resol=256, batch_size=32):
     
    # load the df with metadata
    df_metadata = pd.read_csv('chexpert/chexpert_metadata.csv')

    # define the data per split
    df_metadata_train = df_metadata[df_metadata['split'] == 0]
    df_metadata_val = df_metadata[df_metadata['split'] == 1]
    df_metadata_test = df_metadata[df_metadata['split'] == 2]

    # load the y
    y_train = torch.from_numpy(df_metadata_train['y'].values)
    y_val = torch.from_numpy(df_metadata_val['y'].values)
    y_test = torch.from_numpy(df_metadata_test['y'].values)

    # load the c
    a_train = torch.from_numpy(df_metadata_train['a'].values)
    a_val = torch.from_numpy(df_metadata_val['a'].values)
    a_test = torch.from_numpy(df_metadata_test['a'].values)
    print('Total size of train, val, test: {}, {}, {}'.format(len(a_train), len(a_val), len(a_test)))

    print('Check: unique values of a_train: (with counts), {}'.format(torch.unique(a_train, return_counts=True)))
    print('Check: unique values of a_val: (with counts), {}'.format(torch.unique(a_val, return_counts=True)))
    print('Check: unique values of a_test: (with counts), {}'.format(torch.unique(a_test, return_counts=True)))

    # define transform functions
    train_transform_func = transforms_chexpert(resol)
    test_transform_func = transforms_chexpert(resol)

    # Define the paths
    train_image_paths = ['{}'.format(df_metadata_train['filename'].iloc[i]) for i in range(len(df_metadata_train))]
    val_image_paths = ['{}'.format(df_metadata_val['filename'].iloc[i]) for i in range(len(df_metadata_val))]
    test_image_paths = ['{}'.format(df_metadata_test['filename'].iloc[i]) for i in range(len(df_metadata_test))]
    print('example of train path: {}'.format(train_image_paths[0]))
    
    # Create dataloaders
    train_loader = create_dataloader(train_image_paths, train_transform_func, batch_size)
    val_loader = create_dataloader(val_image_paths, test_transform_func, batch_size)
    test_loader = create_dataloader(test_image_paths, test_transform_func, batch_size)




    # set filename
    filename = os.path.join(save_dir, 'data_chexpert.h5') 

    print("Saving data to: {}".format(filename))




    with h5py.File(filename, 'w') as f:
        # Save metadata
        f.create_dataset('a_train', data=a_train.numpy())
        f.create_dataset('y_train', data=y_train.numpy())
        f.create_dataset('a_val', data=a_val.numpy())
        f.create_dataset('y_val', data=y_val.numpy())
        f.create_dataset('a_test', data=a_test.numpy())
        f.create_dataset('y_test', data=y_test.numpy())

    print("Metadata saved successfully!")
    
    # Save images in parallel
    save_dataset_parallel(filename, 'X_train', train_loader, chunk_dim=batch_size)
    save_dataset_parallel(filename, 'X_val', val_loader, chunk_dim=batch_size)
    save_dataset_parallel(filename, 'X_test', test_loader, chunk_dim=batch_size)

    print("Data saved successfully!")


if __name__ == "__main__":
    main()
   

  