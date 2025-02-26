
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:06:08 2022

"""

# standard libraries 
import os
import numpy as np
import random
import pandas as pd
import torch

# for images
from PIL import Image
from tqdm import tqdm

import argparse
import os


def main(main_task_name, concept_name, folder='CelebA'):

    # read the dataframes
    df_att = pd.read_csv(folder + '/list_attr_celeba.txt', sep='\s+', skiprows=1)
    df_eval = pd.read_csv(folder + '/list_eval_partition.txt', sep='\s+', skiprows=0, header=None)
    df_eval.columns = ['image_id', 'split']
    df_att = df_att.reset_index(drop=False)
    df_att.columns = ['image_id'] + list(df_att.columns[1:])

    # Add  a Female column by replacing -1 with 1 and vice versa
    df_att['Female'] = df_att['Male']*-1

    # Add a y column for the main task
    df_att['y'] = df_att[main_task_name].replace({-1:0})

    # merge the two dataframes on image_id
    df_metadata = pd.merge(df_att, df_eval, on='image_id')
    df_metadata['img_filename'] = df_metadata['image_id']

    # get the main-task and concept labels. Turn these to (1,0) instead of (-1,1)
    # add the columns in the dataframe
    main_task_name = main_task_name
    concept_name = concept_name
    y_m = torch.tensor(df_metadata[main_task_name].replace({-1:0}).values)
    y_c = torch.tensor(df_metadata[concept_name].replace({-1:0}).values)
    df_metadata['y_m'] = y_m
    df_metadata['y_c'] = y_c

    # save the metadata
    df_metadata.to_csv(folder + '_metadata_celeba_{}_{}.csv'.format(main_task_name, concept_name), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--main_task_name', help ='Name of the main task')
    parser.add_argument('--concept_name', help ='Name of the concept')
    args = parser.parse_args()
    dict_arguments = vars(args)

  
    main(args.main_task_name, args.concept_name)
  