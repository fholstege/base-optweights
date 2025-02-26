import h5py
import numpy as np
import argparse
import multiprocessing as mp
import tempfile


def main(dataset_file, train_val_split, batch_size=128, compression="gzip", compression_opts=4, chunk_size =(128, 3, 224, 224)):
    # Open the h5 file
    with h5py.File(dataset_file, 'r') as h5_file:
        X_train = h5_file['X_train']
        y_train = h5_file['y_train']
        c_train = h5_file['c_train']
        X_val = h5_file['X_val']
        y_val = h5_file['y_val']
        c_val = h5_file['c_val']


        print('p(y=1 in old val): {}'.format(np.mean(y_val)))
        print('p(c=1 in old val): {}'.format(np.mean(c_val)))

        # Calculate the new number of training and validation samples
        total_samples = len(X_train) + len(X_val)
        new_train_size = int(total_samples * train_val_split)
        new_val_size = total_samples - new_train_size
        print('New train size: {}'.format(new_train_size))
        print('New val size: {}'.format(new_val_size))

        # define the filename based on the split
        filename = dataset_file.split('.')[0]
        train_val_split_str = str(train_val_split).replace('.', '')
        dataset_file = filename + '_train_val_split_{}.h5'.format(train_val_split_str)


        # Create new datasets in a new .h5 file
        with h5py.File(dataset_file, 'w') as new_h5_file:

            # define the dimensionality of the datasets
            new_X_train_dims = (new_train_size,) + X_train.shape[1:]
            new_y_train_dims = (new_train_size,) + y_train.shape[1:]
            new_c_train_dims = (new_train_size,) + c_train.shape[1:]
            new_X_val_dims = (new_val_size,) + X_val.shape[1:]
            new_y_val_dims = (new_val_size,) + y_val.shape[1:]
            new_c_val_dims = (new_val_size,) + c_val.shape[1:]
            print('New X train dims: {}'.format(new_X_train_dims))
            print('New y train dims: {}'.format(new_y_train_dims))
            print('New c train dims: {}'.format(new_c_train_dims))
            print('New X val dims: {}'.format(new_X_val_dims))
            print('New y val dims: {}'.format(new_y_val_dims))
            print('New c val dims: {}'.format(new_c_val_dims))


            # Create the datasets
            new_X_train = new_h5_file.create_dataset('X_train', new_X_train_dims, dtype='float32', compression=compression, compression_opts=compression_opts, chunks=chunk_size)
            new_y_train = new_h5_file.create_dataset('y_train', new_y_train_dims, dtype='float32', compression=compression, compression_opts=compression_opts, chunks=(chunk_size[0],))
            new_c_train = new_h5_file.create_dataset('c_train', new_c_train_dims, dtype='float32', compression=compression, compression_opts=compression_opts, chunks=(chunk_size[0],))
            new_X_val = new_h5_file.create_dataset('X_val', new_X_val_dims, dtype='float32', compression=compression, compression_opts=compression_opts, chunks=chunk_size)
            new_y_val = new_h5_file.create_dataset('y_val', new_y_val_dims, dtype='float32', compression=compression, compression_opts=compression_opts, chunks=(chunk_size[0],))
            new_c_val = new_h5_file.create_dataset('c_val', new_c_val_dims, dtype='float32', compression=compression, compression_opts=compression_opts, chunks=(chunk_size[0],))

            # Adjust the datasets
            if new_train_size > len(X_train):

                # Move data from validation to training
                num_to_move = new_train_size - len(X_train)

                # First, store the old train data
                print('Storing old training data')
                for i in range(0, len(X_train), batch_size):
                    print('At batch {}/{}'.format(i, len(X_train)))
                    end = min(i + batch_size, len(X_train))
                    new_X_train[i:end] = X_train[i:end]
                    new_y_train[i:end] = y_train[i:end]
                    new_c_train[i:end] = c_train[i:end]
              
                # Second, add the observations from the validation set
                print('Adding validation data to training')
                for i in range(0, num_to_move, batch_size):
                    print('At batch {}/{}'.format(i, num_to_move))
                    end = min(i + batch_size, num_to_move)
                    new_X_train[(len(X_train)+i):(len(X_train)+end)] = X_val[i:end]
                    new_y_train[(len(y_train)+i):(len(y_train)+end)] = y_val[i:end]
                    new_c_train[(len(c_train)+i):(len(c_train)+end)] = c_val[i:end]


                # Third, store the remaining validation data
                print('Storing remaining validation data')
                i_val = 0
                for i in range(num_to_move, len(X_val), batch_size):
                    end = min(i + batch_size, len(X_val))
                    end_val = min(i_val + batch_size, new_val_size)
                    new_X_val[i_val:end_val] = X_val[i:end]
                    new_y_val[i_val:end_val] = y_val[i:end]
                    new_c_val[i_val:end_val] = c_val[i:end]
                    i_val += batch_size


                

              
            else:
                # Move data from training to validation
                num_to_move = len(X_train) - new_train_size

                # First, store the old validation data
                print('Storing old validation data')
                 # Store the old validation data
                for i in range(0, len(X_val), batch_size):
                    print('At batch {}/{}'.format(i, len(X_val)))
                    end = min(i + batch_size, len(X_val))
                    new_X_val[i:end] = X_val[i:end]
                    new_y_val[i:end] = y_val[i:end]
                    new_c_val[i:end] = c_val[i:end]

                # Add the observations from the training set
                for i in range(0, num_to_move, batch_size):
                    print('At batch {}/{}'.format(i, num_to_move))
                    end = min(i + batch_size, num_to_move)
                    new_X_val[len(X_val) + i:len(X_val) + end] = X_train[i:end]
                    new_y_val[len(y_val) + i:len(y_val) + end] = y_train[i:end]
                    new_c_val[len(c_val) + i:len(c_val) + end] = c_train[i:end]

                # Store the remaining training data
                for i in range(num_to_move, len(X_train), batch_size):
                    print('At batch {}/{}'.format(i, len(X_train)))
                    end = min(i + batch_size, len(X_train))
                    new_X_train[i - num_to_move:end - num_to_move] = X_train[i:end]
                    new_y_train[i - num_to_move:end - num_to_move] = y_train[i:end]
                    new_c_train[i - num_to_move:end - num_to_move] = c_train[i:end]
                    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset preparation')
    parser.add_argument('--dataset_file', type=str, help='The dataset file')
    parser.add_argument('--train_val_split', type=float, help='The train-val split')
    args = parser.parse_args()

    # Run the main function
    main(args.dataset_file, args.train_val_split)