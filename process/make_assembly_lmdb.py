import pandas as pd
import os
import sys
sys.path.append('.')
from utils.dataset import LMDBDatabase


def df_to_lmdb_table(df):
    split_catalog_table = {}
    for split, df_split in df.groupby('split'):
        print('Split:', split)
        split_catalog_table[split] = {'all_dbs': []}
        for db_name, df_task_db in df_split.groupby('db'):
            split_catalog_table[split]['all_dbs'].append(db_name)
            this_table = {'-'.join([db_name, str(i)]): data_id for
                          i, data_id in enumerate(df_task_db['data_id'].values.astype('str'))}
            this_table[db_name] = len(this_table)
            split_catalog_table[split].update(this_table)
            print(f"- ({db_name}): {this_table[db_name]}")
    return split_catalog_table


if __name__ == '__main__':
    ASS_DIR = 'data_train/assemblies'
    SPLIT_NAME = 'assembly'
    df_split = pd.read_csv(os.path.join(ASS_DIR, 'split_train_val.csv'))

    lmdb_table = df_to_lmdb_table(df_split)
    print('train dbs:', lmdb_table['train']['all_dbs'])
    print('size train/val:', len(lmdb_table['train']), len(lmdb_table['val']))

    for split in lmdb_table.keys():
        lmdb = LMDBDatabase(os.path.join(ASS_DIR, f'split_{SPLIT_NAME}_{split}.lmdb'), readonly=False)
        lmdb.add(lmdb_table[split])
