# several standard functions that are used in the main function
import os

# import the data, model, and helper functions
from data import WB, CelebA, multiNLI, chexpert
from model import resnet_model, embedding_creator_resnet, get_embedding_in_batches_images, get_embedding_in_batches_tokens, embedding_creator_BERT, BERT_model
from helpers import set_seed, str_to_bool

# import the necessary libraries
import torch
import argparse
import sys

def main(dataset, dataset_file, model_name, model_folder, device_type, seed, workers=0, y_dim=1, batch_size=128, include_test=True):


    # set the device
    device = torch.device(device_type)

    # determine the data_obj
    if dataset == 'WB':
        data_obj = WB()
        data = data_obj.load_data(dataset_file)

        # create group variables
        g_train = data_obj.create_g(data['y_train'], data['c_train'])
        g_val = data_obj.create_g(data['y_val'], data['c_val'])
        g_test = data_obj.create_g(data['y_test'], data['c_test'])

        # set the data attributes
        data_obj.set_data_attributes(data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'],  data['y_test'], device)

        # create the loaders
        set_seed(seed)
        data_obj.create_loaders(batch_size, workers, shuffle=False, include_weights=False, train_weights = None, val_weights = None, pin_memory=True, include_test=include_test)
    
    elif dataset == 'CelebA':

        data_obj = CelebA()

        # load the y and c values
        if include_test:
            y_train, c_train, y_val, c_val, y_test, c_test = data_obj.load_y_c(dataset_file, include_test=include_test)
        else:
            y_train, c_train, y_val, c_val = data_obj.load_y_c(dataset_file, include_test=include_test)
            y_test = None
            c_test = None

        # create group variables
        g_train = data_obj.create_g(y_train, c_train)
        g_val = data_obj.create_g(y_val, c_val)

        if include_test:
            g_test = data_obj.create_g(y_test, c_test)
            data_obj.y_test = y_test
            data_obj.g_test = g_test

        # set the y & g values of the data object
        data_obj.y_train = y_train
        data_obj.y_val = y_val
        data_obj.g_train = g_train
        data_obj.g_val = g_val


        # create the loaders
        data_obj.create_loaders(batch_size, workers, shuffle=False, pin_memory=True, h5_file_path=dataset_file, 
                        x_key_train='X_train', y_key_train='y_train', x_key_val='X_val', y_key_val='y_val', x_key_test='X_test', y_key_test='y_test',
                          device=device, include_test=include_test)
        
    elif dataset == 'chexpert':

        data_obj = chexpert()

        # load the y, set attribute as groups
        y_train, g_train, y_val, g_val, y_test, g_test = data_obj.load_y_a(dataset_file)

        # set the y & g values of the data object
        data_obj.y_train = y_train
        data_obj.y_val = y_val
        data_obj.y_test = y_test
        data_obj.g_train = g_train
        data_obj.g_val = g_val
        data_obj.g_test = g_test


        # create the loaders
        data_obj.create_loaders(batch_size, workers, shuffle=False, pin_memory=True, h5_file_path=dataset_file,
                                x_key_train='X_train', y_key_train='y_train', x_key_val='X_val', y_key_val='y_val', x_key_test='X_test', y_key_test='y_test',
                                device=device, include_test=include_test)
    
    elif dataset == 'multiNLI':
        data_obj = multiNLI()
        data_obj.load_tokens('data/multiNLI')

        g_train = data_obj.g_train
        g_val = data_obj.g_val
        g_test =  data_obj.g_test
    
        data_obj.create_loaders(batch_size, shuffle=False, workers=workers, pin_memory=True, include_test=True)
    
    # get the model
    if dataset == 'WB':
        model_file_name = '{}/WB_model_seed_{}.pt'.format(model_folder, seed)
    elif dataset == 'chexpert':
        model_file_name = '{}/chexpert_model_seed_{}.pt'.format(model_folder, seed)
    elif dataset == 'CelebA':
        model_file_name = '{}/CelebA_model_seed_{}.pt'.format(model_folder, seed)
    elif dataset == 'multiNLI':
        model_file_name = '{}/multiNLI_model_seed_{}.pt'.format(model_folder, seed)

    print('model_file_name: ', model_file_name)


    # check if the model exists
    if os.path.exists(model_file_name):
        if dataset == 'WB' or dataset == 'CelebA' or dataset == 'chexpert':
            model_obj = resnet_model(model_name, y_dim)
        elif dataset == 'multiNLI':
            model_obj = BERT_model(y_dim, replace_fc=False, output_hidden_states=True, output_attentions=False, embedding_creator_forward=True)
        model_obj.load_state_dict(torch.load( model_file_name, map_location=device))
        model_obj.to(device)
    else:
        # throw an error if the model does not exist
        raise ValueError('The model does not exist')


    # from the trained model, get the classifier
    if dataset == 'multiNLI':
        classifier = model_obj.model.classifier
      
    
    # create the embedding creator
    if dataset == 'WB' or dataset == 'CelebA' or dataset == 'chexpert':
        embedding_creator_obj = embedding_creator_resnet(model_obj.base_model, device)
    elif dataset == 'multiNLI':
        embedding_creator_obj = embedding_creator_BERT(model_obj, embedding_type='pool')

    # create the loaders for the embeddings
    train_loader = data_obj.dict_loaders['train']
    val_loader = data_obj.dict_loaders['val']
    if include_test:
        test_loader = data_obj.dict_loaders['test']


    # get the embeddings
    if dataset == 'WB' or dataset == 'CelebA' or dataset == 'chexpert':
        to_float32 = True if dataset == 'CelebA' else False

        train_embeddings = get_embedding_in_batches_images( embedding_creator_obj, train_loader, to_float32=to_float32)
        val_embeddings = get_embedding_in_batches_images( embedding_creator_obj, val_loader, to_float32=to_float32)

        # if include_test, get the test embeddings
        if include_test:
            test_embeddings = get_embedding_in_batches_images( embedding_creator_obj, test_loader, to_float32=to_float32)
    elif dataset == 'multiNLI':
        train_embeddings = get_embedding_in_batches_tokens( embedding_creator_obj, train_loader, device, save_dir='./temp_embeddings_{}'.format(seed), classifier=classifier)
        val_embeddings = get_embedding_in_batches_tokens( embedding_creator_obj, val_loader, device, save_dir='./temp_embeddings_{}'.format(seed), classifier=classifier)
        if include_test:
            test_embeddings = get_embedding_in_batches_tokens( embedding_creator_obj, test_loader, device, save_dir='./temp_embeddings_{}'.format(seed), classifier=classifier)
      

    # save the embeddings in a folder
    # the folder corresponds to the embeddings/model_file_name
    file_for_embeddings = 'embeddings/' + model_file_name.replace('.pt', '')
    if not os.path.exists(file_for_embeddings):
        os.makedirs(file_for_embeddings)


    # create a dictionary to save the embeddings, y values, and g values
    data = {'X_train': train_embeddings, 'X_val': val_embeddings,
             'y_train': data_obj.y_train, 'y_val': data_obj.y_val,
             'g_train': g_train, 'g_val': g_val}
    
    if include_test:
        data['X_test'] = test_embeddings
        data['y_test'] = data_obj.y_test
        data['g_test'] = g_test
    
    # save the embeddings in .pt files
    torch.save(data, 'embeddings/' + model_file_name.replace('.pt', '') + '/data.pt')
    print('Just saved at: embeddings/' + model_file_name.replace('.pt', ''))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset preparation')
    parser.add_argument('--dataset', type=str, help='The dataset to use')
    parser.add_argument('--dataset_file', type=str, help='The .pkl file containing the dataset')
    parser.add_argument('--model_name', type=str, default='resnet50', help='The name of the model to use')
    parser.add_argument('--model_folder', type=str, default='', help='The folder to save the model')
    parser.add_argument('--device_type', type=str, default='cuda', help='The type of device to use')
    parser.add_argument('--include_test', type=str, default='True', help='Whether to include the test set')
    parser.add_argument('--seed', type=int, default=42, help='The random seed to use')


    args = parser.parse_args()
    
    args.include_test = str_to_bool(args.include_test)
 
   

    # Run the main function
    main(args.dataset, args.dataset_file, args.model_name, args.model_folder,  args.device_type, args.seed, include_test=args.include_test)

   

