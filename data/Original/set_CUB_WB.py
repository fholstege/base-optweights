
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:06:08 2022

"""

# standard libraries 
import os
import numpy as np
import random
import pandas as pd


# for images
from PIL import Image
from tqdm import tqdm

import argparse
import os




def crop_and_resize(source_img, target_img):
    """
    Make source_img exactly the same as target_img by expanding/shrinking and
    cropping appropriately.
    If source_img's dimensions are strictly greater than or equal to the
    corresponding target img dimensions, we crop left/right or top/bottom
    depending on aspect ratio, then shrink down.
    If any of source img's dimensions are smaller than target img's dimensions,
    we expand the source img and then crop accordingly
    Modified from
    https://stackoverflow.com/questions/4744372/reducing-the-width-height-of-an-image-to-fit-a-given-aspect-ratio-how-python
    """
    source_width = source_img.size[0]
    source_height = source_img.size[1]

    target_width = target_img.size[0]
    target_height = target_img.size[1]

    # Check if source does not completely cover target
    if (source_width < target_width) or (source_height < target_height):
        # Try matching width
        width_resize = (target_width, int((target_width / source_width) * source_height))
        if (width_resize[0] >= target_width) and (width_resize[1] >= target_height):
            source_resized = source_img.resize(width_resize, Image.LANCZOS)
        else:
            height_resize = (int((target_height / source_height) * source_width), target_height)
            assert (height_resize[0] >= target_width) and (height_resize[1] >= target_height)
            source_resized = source_img.resize(height_resize, Image.LANCZOS)
        # Rerun the cropping
        return crop_and_resize(source_resized, target_img)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height

    if source_aspect > target_aspect:
        # Crop left/right
        new_source_width = int(target_aspect * source_height)
        offset = (source_width - new_source_width) // 2
        resize = (offset, 0, source_width - offset, source_height)
    else:
        # Crop top/bottom
        new_source_height = int(source_width / target_aspect)
        offset = (source_height - new_source_height) // 2
        resize = (0, offset, source_width, source_height - offset)

    source_resized = source_img.crop(resize).resize((target_width, target_height), Image.LANCZOS)
    return source_resized

def combine_and_mask(img_new, mask, img_black):
    """
    Combine img_new, mask, and image_black based on the mask
    img_new: new (unmasked image)
    mask: binary mask of bird image
    img_black: already-masked bird image (bird only)
    """
    # Warp new img to match black img
    img_resized = crop_and_resize(img_new, img_black)
    img_resized_np = np.asarray(img_resized)

    # Mask new img
    img_masked_np = np.around(img_resized_np * (1 - mask)).astype(np.uint8)

    # Combine
    img_combined_np = np.asarray(img_black) + img_masked_np
    img_combined = Image.fromarray(img_combined_np)

    return img_combined


def main(spurious_corr_values,  val_frac=0.2, add_images=True):

    # here we store the place data 
    place_dir = 'WB_data/water_land'

    # here we store the image data
    for spurious_corr in spurious_corr_values:
        print('Working on: {}'.format(spurious_corr))
        
        # specify the filename metadata
        filename_metadata = 'metadata_waterbird_{}_{}_50.csv'.format(str(int(spurious_corr*100)), str(int(spurious_corr*100)))

        # specify the folder name
        folder_name = 'waterbirds_'+str(int(spurious_corr*100))

        # Check whether the specified path exists or not
        folder_exists = os.path.exists('WB_data/images/'+ folder_name)
        if not folder_exists:
            # Create a new directory because it does not exist
            os.makedirs('WB_data/images/'+ folder_name)

        # specify the output folder
        output_dir = 'WB_data/images/'+folder_name

        ##################
        # 1. Load in the meta-data on the CUB dataset, 
        # determine which birds are waterbirds and others are landbirds
        ##################

        # get overview of all the files
        image_text_file_path = 'WB_data/images.txt'
        df_images = pd.read_csv(image_text_file_path, # path to images 
                                    sep = " ", # separate the files
                                    header = None, # no header
                                    names = ['img_id', 'img_filename'], # set column names
                                    index_col = 'img_id')
        

         # get a list of all the bird species
        all_bird_species = [img_filename.split('/')[0].split('.')[1].lower() for img_filename in df_images['img_filename']]

        # get a unique list of all the bird species
        unique_bird_species = np.unique(all_bird_species)

        # all the species that are waterbirds
        water_birds_list = [
            'Albatross', # Seabirds
            'Auklet',
            'Cormorant',
            'Frigatebird',
            'Fulmar',
            'Gull',
            'Jaeger',
            'Kittiwake',
            'Pelican',
            'Puffin',
            'Tern',
            'Gadwall', # Waterfowl
            'Grebe',
            'Mallard',
            'Merganser',
            'Guillemot',
            'Pacific_Loon'
        ]

        # dict that saves per species if waterbird or not
        water_birds = {}

        # go over each species
        for species in unique_bird_species:
            water_birds[species] = 0 # standard; 0 (not water bird)
            for water_bird in water_birds_list: # go over the water birds
                if water_bird.lower() in species: 
                    water_birds[species] = 1 # set if water bird in species

        # add variable with 1 if water bird, 0 if not water bird
        df_images['y'] = [water_birds[species] for species in all_bird_species]

        ##################
        # 2. Determine the train, validation and test split
        # and the percentage of backgrounds for each
        ##################

        # save a dataframe with the training and test split
        train_test_df =  pd.read_csv(
        'WB_data/train_test_split.txt',
            sep=" ",
            header=None,
            names=['img_id', 'split'],
            index_col='img_id')

        # add column with image id added to it
        df_images = df_images.join(train_test_df, on='img_id')

        # acquire test, train and validation id
        test_ids = df_images.loc[df_images['split'] == 0].index
        train_ids = np.array(df_images.loc[df_images['split'] == 1].index)
        val_ids = np.random.choice(
            train_ids,
            size=int(np.round(val_frac * len(train_ids))),
            replace=False)

        # set the split id
        df_images.loc[train_ids, 'split'] = 0
        df_images.loc[val_ids, 'split'] = 1
        df_images.loc[test_ids, 'split'] = 2

        # standard value of place is zero
        df_images['place'] = 0

        # train, validation and test ids
        train_ids = np.array(df_images.loc[df_images['split'] == 0].index)
        val_ids = np.array(df_images.loc[df_images['split'] == 1].index)
        test_ids = np.array(df_images.loc[df_images['split'] == 2].index)

        # go over (1) type of split and (2) ids in that type of split
        for split_idx, ids in enumerate([train_ids, val_ids, test_ids]):
            for y in (0, 1): # go over cases; waterbird, landbird
            
                if split_idx == 0 or split_idx == 1: # train and validation
                
                    # set likelihood of appearing in corresponding background
                    if y == 0:
                        pos_fraction = 1 - spurious_corr # if land bird, 1- spurious_corr chance of having a water background
                    else:
                        pos_fraction = spurious_corr # if waterd bird, spurious_corr chance of having a water background
                
                # if test set (split_idx == 2), 50/50
                else:
                    pos_fraction = 0.5
                
                # df for the split
                subset_df = df_images.loc[ids, :]
                
                # y values for this split
                y_ids = np.array((subset_df.loc[subset_df['y'] == y]).index)
                
                # ids of position place
                pos_place_ids = np.random.choice(
                    y_ids,
                    size=int(np.round(pos_fraction * len(y_ids))),
                    replace=False)
                
                # set the ids where place is 1
                df_images.loc[pos_place_ids, 'place'] = 1

        ##################
        # 3. assign to each bird type and place combination an image file
        ##################


        # which places to add
        target_places = [
            ['bamboo_forest', 'forest-broadleaf'],  # Land backgrounds
            ['ocean', 'lake-natural']]              # Water backgrounds


        # check; training, validation and test distribution 
        for split, split_label in [(0, 'train'), (1, 'val'), (2, 'test')]:
            print(f"{split_label}:")
            split_df = df_images.loc[df_images['split'] == split, :]
            print(f"waterbirds are {np.mean(split_df['y']):.3f} of the examples")
            print(f"y = 0, c = 0: {np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 0):.3f}, n = {np.sum((split_df['y'] == 0) & (split_df['place'] == 0))}")
            print(f"y = 0, c = 1: {np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 1):.3f}, n = {np.sum((split_df['y'] == 0) & (split_df['place'] == 1))}")
            print(f"y = 1, c = 0: {np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 0):.3f}, n = {np.sum((split_df['y'] == 1) & (split_df['place'] == 0))}")
            print(f"y = 1, c = 1: {np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 1):.3f}, n = {np.sum((split_df['y'] == 1) & (split_df['place'] == 1))}")


        # folder belonging to each place
        place_ids_df = pd.read_csv(
            place_dir+ '/categories_places365.txt',
            sep=" ",
            header=None,
            names=['place_name', 'place_id'],
            index_col='place_id')

        # list with target place ids
        target_place_ids = []


        # go over [bamboo_forest, forest_broadleaf] and [ocean, lake_natural]
        for idx, target_places_category in enumerate(target_places):
            place_filenames = []
            print(f'category {idx} {target_places_category}')
                
            # go over each place type
            for target_place in target_places_category:
                target_place_ids.append(target_place)   
                print(f'category {idx} {target_place} has id {target_place_ids[idx]}')
                    
                                    # get filename of places 
                place_filenames += [
                        f'{target_place}/{filename}' for filename in os.listdir(
                            os.path.join(place_dir,target_place))
                        if filename.endswith('.jpg')]
                
                print(f'number of files in {target_place}: {len(place_filenames)}')
                
            # shuffle the place filenames 
            random.shuffle(place_filenames)

            # Assign each filename to an image
            indices = (df_images.loc[:, 'place'] == idx)
            print('first 5 elements of indices', indices[:5])
            assert len(place_filenames) >= np.sum(indices),\
                f"Not enough places ({len(place_filenames)}) to fit the dataset ({np.sum(df_images.loc[:, 'place'] == idx)})"
            df_images.loc[indices, 'place_filename'] = place_filenames[:np.sum(indices)]

            print(f"Assigned {np.sum(indices)} images to category {idx} {target_places_category}")


        print(df_images.head())
        print(df_images['place_filename'].head())

        df_images_train = df_images.loc[df_images['split'] == 0, :]
        df_images_train_check = df_images_train.groupby(['y', 'place']).size()
        print(df_images_train_check.shape)
        print(df_images_train_check)



        ###################
        # 4. Combine filenames 
        ###################

        image_path ='CUB_data/CUB_200_2011/images'
        segmentation_path = 'WB_data/segmentations'

        for i in tqdm(df_images.index):
            
            # image of the bird
            img_path = os.path.join(image_path, df_images.loc[i, 'img_filename'])
            
            # get the image of the segmenation
            seg_path = os.path.join(segmentation_path, df_images.loc[i, 'img_filename'].replace('.jpg', '.png'))
            
            # get the place path 
            place_path = os.path.join(place_dir, df_images.loc[i, 'place_filename'])
            place = Image.open(place_path).convert('RGB')
            
            # set images to numpy
            img_np = np.asarray(Image.open(img_path).convert('RGB'))
            seg_np = np.asarray(Image.open(seg_path).convert('RGB'))/255
                
            # create image of bird with black background
            img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
            
            # create image of bird with place background
            combined_img = combine_and_mask(place, seg_np, img_black)
            
            # create image of bird only
            bird_img = combine_and_mask(Image.fromarray(np.ones_like(place) * 150), seg_np, img_black)
            
            # select image of place
            seg_np *= 0.
            img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
            place_img = combine_and_mask(place, seg_np * 0, img_black)

            # call the path for each type of data
            combined_path = os.path.join(output_dir, "combined", df_images.loc[i, 'img_filename'])
            os.makedirs('/'.join(combined_path.split('/')[:-1]), exist_ok=True)
            combined_img.save(combined_path)


            # save images to folder
            if add_images:
                bird_path = os.path.join(output_dir, "birds", df_images.loc[i, 'img_filename'])
                place_path = os.path.join(output_dir, "places", df_images.loc[i, 'img_filename'])
                
                # make directory
                os.makedirs('/'.join(bird_path.split('/')[:-1]), exist_ok=True)
                os.makedirs('/'.join(place_path.split('/')[:-1]), exist_ok=True)
            
                # save images to folder 
                bird_img.save(bird_path)
                place_img.save(place_path)

        # write csv of metadata
        df_images.to_csv(os.path.join(output_dir, filename_metadata))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--spurious_corr_values', help ='list of confounder strength values')
    args = parser.parse_args()
    dict_arguments = vars(args)

    string_spurious_corr= dict_arguments['spurious_corr_values']
    
    list_spurious_corr = list(string_spurious_corr.split("-"))
    list_spurious_corr = [float(x) for x in list_spurious_corr]


  
    spurious_corr_values = list_spurious_corr

    main(spurious_corr_values)
  