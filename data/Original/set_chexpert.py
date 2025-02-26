


import os
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys




def main( test_pct=0.15, val_pct=0.1):

    # define the directory
    chexpert_dir = 'chexpert'

    # load the data on attributes
    meta_file = os.path.join(chexpert_dir,'CHEXPERT DEMO.xlsx')
    df_att = pd.read_excel(meta_file, engine='openpyxl')[['PATIENT', 'PRIMARY_RACE']]

    # load train and valid csv
    train_file = os.path.join(chexpert_dir,'train.csv')
    val_file = os.path.join(chexpert_dir,'valid.csv')
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df = pd.concat([df_train, df_val], ignore_index=True)

    # define the filename
    df['filename'] = df['Path'].astype(str).apply(lambda x: os.path.join(chexpert_dir, x[x.index('/')+1:]))

    # define the subject id
    df['subject_id'] = df['Path'].apply(lambda x: int(Path(x).parent.parent.name[7:])).astype(str)

    # ensure in either Male or Female
    df = df[df.Sex.isin(['Male', 'Female'])]

    # for meta data, define subject id
    df_att['subject_id'] = df_att['PATIENT'].apply(lambda x: x[7:]).astype(int).astype(str)

    
    # merge the data
    df = pd.merge(df, df_att, on='subject_id', how='inner').reset_index(drop=True)


    def cat_race(r):
        if isinstance(r, str):
            if r.startswith('White'):
                return 0
            elif r.startswith('Black'):
                return 1
        return 2

    # define the attributes
    df['ethnicity'] = df['PRIMARY_RACE'].apply(cat_race)
    attr_mapping = {'Male_0': 0, 'Female_0': 1, 'Male_1': 2, 'Female_1': 3, 'Male_2': 4, 'Female_2': 5}
    df['a'] = (df['Sex'] + '_' + df['ethnicity'].astype(str)).map(attr_mapping)
    df['y'] = df['No Finding'].fillna(0.0).astype(int)

    # define the splitj
    train_val_idx, test_idx = train_test_split(df.index, test_size=test_pct, random_state=42, stratify=df['a'])
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_pct/(1-test_pct), random_state=42, stratify=df.loc[train_val_idx, 'a'])

    df['split'] = 0
    df.loc[val_idx, 'split'] = 1
    df.loc[test_idx, 'split'] = 2


    # define the metadata filename
    df_meta_filename = os.path.join(chexpert_dir, 'chexpert_metadata.csv')
    df.to_csv(df_meta_filename, index=False)








    




   



if __name__ == "__main__":


    # Run the main function
    main()
