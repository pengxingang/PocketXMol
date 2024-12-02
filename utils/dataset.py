from itertools import cycle
import os
from typing import Iterator
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
import lmdb
from torch.utils.data import get_worker_info
from torch.distributed import get_rank, get_world_size

from torch.utils.data import Dataset, Sampler, IterableDataset
from torch_geometric.data import Batch
try:
    from .train import shuffled_cyclic_iterator
except:
    import sys
    sys.path.append('.')
    from utils.train import shuffled_cyclic_iterator

LMDB_CONFIGS = {  # seems not used
    'geom': ['mols', 'torsion', 'decom'],
    'qm9': ['mols', 'torsion', 'decom'],
    'unmi': ['mols', 'torsion', 'decom'],
    'csd': ['pocmol10', 'torsion', 'decom'],
    'pbdock': ['pocmol10', 'torsion', 'decom'],
    'moad': ['pocmol10', 'torsion', 'decom'],
    'apep': ['pocmol10', 'torsion', 'decom'],

    'cremp': ['mols'],
    'pepbdb': ['pocmol10'],
}

    
def make_split(df_dict, provided_test=[], test_size_dict=None,
    ratio_val=0.1, ratio_test = 0.1,
    max_val=10000, max_test=10000):
    test_size_dict = test_size_dict if test_size_dict is not None else {}
    
    # find the task with the minimum number of data_id first
    df_num = pd.DataFrame(None, index=[task for task in df_dict.keys()],
                          columns=['train_size', 'train_ratio', 'val_size',
                                   'val_ratio', 'test_size', 'test_ratio'])
    df_num['total'] = [df.shape[0] for df in df_dict.values()]
    
    sorted_tasks = [key for key, value in sorted(
        df_dict.items(), key=lambda item: item[1]['data_id'].unique().shape[0])]
    # test set
    test_ids = set(provided_test)
    for task in sorted_tasks:
        df = df_dict[task]
        data_id_counts = df['data_id'].value_counts()
        if task in test_size_dict.keys():
            num_data_test = test_size_dict[task]
        else:
            num_data_test =  min(int(data_id_counts.shape[0] * ratio_test), max_test)
        if len(test_ids) < num_data_test: # no enough data for test
            # add more data to test
            data_id_counts_remain = data_id_counts[~data_id_counts.index.isin(test_ids)]
            num_data_test_reamin = int(num_data_test - len(test_ids))
            test_ids_this = np.random.choice(data_id_counts_remain.index.values,
                        p=data_id_counts_remain.values/np.sum(data_id_counts_remain.values),
                        size=num_data_test_reamin, replace=False)
            test_ids.update(test_ids_this)

    df_num['test_size'] = [df_dict[task]['data_id'].isin(test_ids).sum() for task in df_dict.keys()]
    df_num['test_ratio'] = df_num['test_size'] / df_num['total']

    # val set
    val_ids = set()
    for task in sorted_tasks:
        df = df_dict[task]
        data_id_counts = df['data_id'].value_counts()
        num_data_val = min(int(data_id_counts.shape[0] * ratio_val), max_val)
        data_id_counts = df[df['data_id'].isin(test_ids) == False]['data_id'].value_counts()
        if len(val_ids) < num_data_val: # no enough data for val
            # add more data to val
            data_id_counts_remain = data_id_counts[~data_id_counts.index.isin(val_ids)]
            num_data_val_remain = int(num_data_val - len(val_ids))
            val_ids_this = np.random.choice(data_id_counts_remain.index.values,
                        p=data_id_counts_remain.values/np.sum(data_id_counts_remain.values),
                        size=num_data_val_remain, replace=False)
            val_ids.update(val_ids_this)
    
    df_num['val_size'] = [df_dict[task]['data_id'].isin(val_ids).sum() for task in df_dict.keys()]
    df_num['val_ratio'] = df_num['val_size'] / df_num['total']
    
    # train set
    train_ids = set()
    for task in sorted_tasks:
        df = df_dict[task]
        train_ids_this = df[~df['data_id'].isin(test_ids | val_ids)]['data_id'].unique()
        train_ids.update(train_ids_this)
    
    df_num['train_size'] = [df_dict[task]['data_id'].isin(train_ids).sum() for task in df_dict.keys()]
    df_num['train_ratio'] = df_num['train_size'] / df_num['total']
    
    # save
    assert train_ids & val_ids & test_ids == set()
    all_ids = list(train_ids) + list(val_ids) + list(test_ids)
    labels = ['train'] * len(train_ids) + ['val'] * len(val_ids) + ['test'] * len(test_ids)
    df_split = pd.DataFrame({'data_id': all_ids, 'split': labels})

    return df_split, df_num
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # df_split.to_csv(save_path, index=False)
    # df_num.to_csv(save_path.replace('.csv', '_num.csv'))
    
    

class RegenDataset(Dataset):
    # make a dataset from previous generated path
    def __init__(self, gen_path, task, file2input, transforms=None):
        super().__init__()
        self.gen_path = gen_path
        self.task = task
        self.file2input = file2input
        self.transforms = transforms
        
        self.gen_name = os.path.basename(gen_path)
        self.df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
        self.sdf_dir = os.path.join(gen_path, f'{self.gen_name}_SDF')
        if not os.path.exists(self.sdf_dir):
            self.sdf_dir = os.path.join(gen_path, 'SDF')
        
        # load pocket_pdb
        pocket_path = os.path.join(self.sdf_dir, 'pocket_block.pdb')
        with open(pocket_path, 'r') as f:
            self.pocket_pdb = f.read()
        
        # load sdf mol list
        df_succ = self.df_gen[self.df_gen['tag'].isna()]
        self.filename_list = df_succ['filename'].values

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        filename = self.filename_list[index]
        data = self.file2input(
            # mol=os.path.join(self.sdf_dir, filename),
            mol=os.path.join(self.sdf_dir, filename.replace('.pdb', '_mol.sdf')),  # pdb may be broken
            pdb=deepcopy(self.pocket_pdb),
            data_id=self.gen_name + '_' + filename,
            pdbid=''
        )
        data.update({'task': self.task,'db':self.gen_name, 'key':''})
        if self.transforms is not None:
            data = self.transforms(data)
        return data



class UseDataset(Dataset):
    def __init__(self, data, n, task='', transforms=None):
        super().__init__()
        self.data = data
        self.n = n
        self.task = task
        self.transforms = transforms
        
    def __len__(self):
        return self.n

    def __getitem__(self, index):
        data = deepcopy(self.data)
        data.update({'task': self.task, 'db':'use', 'key': ''})
        if self.transforms is not None:
            data = self.transforms(data)
        return data

class TestTaskDataset(Dataset):
    """Transfor the ForeverTaskDataset to a normal dataset for testing purpose"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.forever_dataset = ForeverTaskDataset(*args, **kwargs)
        self.size = len(self)
        
    def __getitem__(self, index):
        if index >= self.size:
            raise IndexError(f'index {index} out of range {len(self)}')
        return self.forever_dataset[index]
    
    def __len__(self):
        return self.forever_dataset.task_db_size(self.forever_dataset.task_dbs[0])
        # return self.forever_dataset.total_size
    


# class MaxSizeDatasetWrapper(IterableDataset):
#     def __init__(self, dataset, max_size,
#                  follow_batch, exclude_keys):
#         super().__init__()
#         self.dataset = dataset
#         self.max_size = max_size
#         self.follow_batch = follow_batch
#         self.exclude_keys = exclude_keys

#     def __iter__(self) -> Iterator:
#         cum_size = 0
#         data_list = []
#         for data in self.dataset:
#             cum_size += data.num_atoms
#             if cum_size > self.max_size:
#                 yield Batch.from_data_list(data_list, follow_batch=self.follow_batch,
#                                            exclude_keys=self.exclude_keys)
#                 cum_size = data.num_atoms
#                 data_list = [data]
#             else:
#                 data_list.append(data)


class ForeverTaskDataset(IterableDataset):
    def __init__(self, dataset_cfg, task_db_weights, mode,
                split=None, transforms=None, shuffle=False, **kwargs):
        super().__init__()
        
        self.dataset_cfg = dataset_cfg
        self.mode = mode
        assert mode in ['train', 'val', 'test'], 'Unknown mode: %s' % mode
        if split is None:
            split = mode
        self.split = split
        self.transforms = transforms
        self.shuffle = shuffle
        
        # task
        if mode in ['train', 'val']:
            self.task_dbs = [(task, db_name) for task, value in task_db_weights.items() for db_name in value['db_ratio']]
            self.task_weights = {
                'tasks': list(task_db_weights.keys()),
                'weights': [value['weight'] for value in task_db_weights.values()]
            }
            self.db_weights = {task: {
                    'dbs': list(value['db_ratio'].keys()),
                    'weights': np.array(list(value['db_ratio'].values())) / sum(value['db_ratio'].values())
                } for task, value in task_db_weights.items()
            }

        else:
            task = task_db_weights['name']
            db_name = task_db_weights['db']
            self.task_dbs = [(task, db_name)]
            

        # path and root
        self.root = dataset_cfg['root']
        self.assembly_path = os.path.join(self.root, dataset_cfg['assembly_path'])
        self.dbs_config = dataset_cfg['dbs']
        # used dbs
        # self.used_dbs = [db.name for db in self.dbs_config]
        
        self.catalog_dict = None
        self.db_dict = None
        # run only once
        self.setup(set_index=True, set_db=True)
        
        # get global id
        if mode != 'test':
            self.gpu_rank = kwargs['global_rank']
            self.world_size = kwargs['world_size']
            self.num_workers = kwargs['num_workers']
            self.setup(set_iter=True)


    def setup(self, set_index=False, set_db=False, set_iter=False,
              ):
        if set_index:
            # setup index
            if self.assembly_path.endswith('.csv'):  # for test only. use the data_id columns
                df_ass = pd.read_csv(self.assembly_path)
                data_id_list = df_ass['data_id'].values
                assert len(self.task_dbs) == 1, 'only test can use csv assembly'
                datatask_db = '-'.join(self.task_dbs[0])
                db_size = len(data_id_list)
                catalog_dict = {
                    'datatask_dbs': [datatask_db],
                    datatask_db: db_size,
                }
                catalog_dict.update({
                    f'{datatask_db}-{i}':data_id_list[i] for i in range(len(data_id_list))
                })
            elif self.assembly_path.endswith('.pkl'):
                # raise NotImplementedError
                assembly_path = self.assembly_path.replace('.pkl', f'_{self.split}.pkl')
                with open(assembly_path, 'rb') as f:
                    catalog_dict = pickle.load(f)
                    # catalog_dict = {task_db:value for task_db, value in catalog_dict.items()}
            elif self.assembly_path.endswith('.lmdb'):
                assembly_path = self.assembly_path.replace('.lmdb', f'_{self.split}.lmdb')
                catalog_dict = LMDBDatabase(assembly_path, max_readers=126)
            else:
                raise ValueError('Unsupported appendix of assembly_path', self.assembly_path)
            # if self.shuffle:
            #     self.catalog_dict = {task_db: np.random.permutation(value) for task_db, value in catalog_dict.items()}
            # else:
            self.catalog_dict = catalog_dict

        if set_db:
            # load db
            db_dict = {}
            for db_config in self.dbs_config:
                db_name = db_config.name
                db_dict[db_name] = SingleDatabase(db_config, self.root)
                # db_dict[db_name] = DB_DICT[db_name](db_config, self.root)
            self.db_dict = db_dict

        if set_iter:
            # global_id, total_samplers = self._get_global_id(self.gpu_rank, self.world_size)

            self.total_samplers = total_samplers = self.num_workers * self.world_size
            samplers_cycler = shuffled_cyclic_iterator(total_samplers, shuffle=self.shuffle)
            # print(self.mode,  total_samplers, id(self.catalog_dict))
            
            sampler_index_list = [{} for _ in range(total_samplers)]
            for db_name in self.catalog_dict['all_dbs']:
                
                # calc basic num per sampler
                db_size = self.catalog_dict[db_name]
                size_per_sampler = np.floor(db_size / total_samplers)

                # who should have a bonus
                n_remaining = db_size % total_samplers
                index_sampler_bonus = [next(samplers_cycler) for _ in range(n_remaining)]
                who_has_bonus = np.zeros(total_samplers, dtype=bool)
                who_has_bonus[index_sampler_bonus] = True
                
                for global_id in range(total_samplers):
                    # calc index for this sampler
                    index_start = int(global_id * size_per_sampler + np.sum(who_has_bonus[:global_id]))
                    index_end = int(index_start + size_per_sampler + int(who_has_bonus[global_id]))
                    sampler_index_list[global_id].update({db_name: (index_start, index_end)})
                    
            self.sampler_index_list = sampler_index_list
            

    def _get_global_id(self):
        # per gpu
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        assert num_workers == self.num_workers
        global_id = self.gpu_rank * self.num_workers + worker_id
        print(f'The processor {global_id}/{self.total_samplers} ({self.gpu_rank}-{worker_id}) is ready for {self.split}!')
        return global_id

    def __iter__(self):
        global_id = self._get_global_id()
        # self.task_inf_iter = {(task, db): shuffled_cyclic_iterator((int(index_start), int(index_end)), shuffle=self.shuffle)
        #                       for task, db, index_start, index_end in self.sampler_index_list[global_id]}
        print('iter from global_id:', global_id)
        sample_index = self.sampler_index_list[global_id]
        if not self.shuffle:
            task_inf_iter = {db_name: cycle(range(*index_range)) 
                             for db_name, index_range in sample_index.items()}
        while True:
            for _ in range(self.total_size//self.total_samplers):
                task = np.random.choice(self.task_weights['tasks'], p=self.task_weights['weights'])
                task_db_weight = self.db_weights[task]
                db = np.random.choice(task_db_weight['dbs'], p=task_db_weight['weights'])
                # get an index
                if self.shuffle:
                    index = np.random.randint(*sample_index[db])
                else:
                    index = next(task_inf_iter[db])
                yield self[(task, db, index)]
            if self.mode != 'train':
                break

            
    def get_data_key(self, task, db_name, data_id):
        
        if task == 'linking':
            data_id_split = data_id.split('/')
            if len(data_id_split) == 2:  # actually it is sep_id
                data_id, index_sep = data_id_split
            else:
                data_id = data_id_split[0]
                index_sep = '0'
        
        # the new process version has merged the torsion and decom
        if db_name in ['geom', 'qm9', 'unmi', ]:
            key = f'mols/{data_id};torsion/{data_id};decom/{data_id}'
        elif db_name in ['csd', 'pbdock', 'pbdockS', 'pbdockL', 'moad']:
            key = f'pocmol10/{data_id};torsion/{data_id};decom/{data_id}'
        elif db_name in ['cremp', 'protacdb']:
            key = f'mols/{data_id}'
        elif db_name in ['apep']:
            key = f'pocmol10/{data_id};torsion/{data_id};decom/{data_id};peptide/{data_id}'
        elif db_name in ['pepbdb', 'qbpep']:
            key = f'pocmol10/{data_id};peptide/{data_id}'
        elif db_name in ['poseb', 'poseboff']:
            key = f'pocmol10/{data_id}'
        else:  # new db. typically for test/use
            # raise ValueError(f'unknown db_name {db_name}')
            key = f'{db_name}/{data_id}'
        
        if task == 'linking':
            key += f';linking/{data_id}/{index_sep}'
        elif task == 'growing':
            key += f';growing/{data_id}'
        return key


    def __getitem__(self, index):
        # select task
        if isinstance(index, tuple):
            if len(index) == 4:
                datatask, task, db_name, idx = index
            else:
                task, db_name, idx = index
        else: # single task mode (usually for test)
            task, db_name = self.task_dbs[0]
            idx = index
        
        # prepare key
        if isinstance(self.catalog_dict, dict):
            key = '-'.join([db_name, str(idx)])
            data_id = self.catalog_dict[key] if key in self.catalog_dict else None
        else:
            data_id = self.catalog_dict['-'.join([db_name, str(idx)])]
        if data_id is None:
            data_id = self.catalog_dict['-'.join([task, db_name, str(idx)])]
            if data_id is None:  # use ranker for non-dock test sets
                db = self.catalog_dict['datatask_dbs'][0]
                data_id = self.catalog_dict['-'.join([db, str(idx)])]

        # select db
        key = self.get_data_key(task, db_name, data_id)
        
        # select data
        data = self.db_dict[db_name][key]
        data.update({'task': task, 'db':db_name, 'key': key})
        if self.transforms is not None:
            data = self.transforms(data)
        #     data, level_dict = self.transforms(data)
        # if True:
        #     import time
        #     import torch
        #     new_data = deepcopy(data)
        #     new_data.update({f'level_{key}':value for key, value in level_dict.items()})
        #     torch.save(new_data, 'data/inputs/0115_pepbdb_1skg_B/{}.pt'.format(time.time()))
        return data
    
    @property
    def total_size(self):
        # if self.catalog_dict is None:
        #     self.setup(set_index=True)
        return sum(self.catalog_dict[db_name] for db_name in self.catalog_dict['all_dbs'])

    # def datatask_db_size(self, datatask_db):
    #     # if self.catalog_dict is None:
    #     #     self.setup(set_index=True)
    #     # return len(self.catalog_dict[datatask_db])
    #     return self.catalog_dict[datatask_db]
    
    def task_db_size(self, task_db):
        task, db = task_db
        size = self.catalog_dict[task+'-'+db]
        if size is None:
            size = self.catalog_dict[db]
            if size is None: # use ranker for non-dock test sets
                db = self.catalog_dict['datatask_dbs'][0]
                size = self.catalog_dict[db]
        return size
        # datatask = self.task_to_datatask[task]
        # return self.datatask_db_size(datatask+'-'+db)



class LMDBDatabase(Dataset):
    def __init__(self, db_path, map_size=int(10e11), readonly=True, **kwargs):
        super().__init__()
        self.db_path = db_path
        self.map_size = map_size
        if readonly:
            self.env = lmdb.open(
                self.db_path,
                map_size=self.map_size,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,  **kwargs
            )
        else:
            self.env = lmdb.open(
                self.db_path,
                map_size=self.map_size,
                subdir=False,
                readonly=False,  **kwargs
            )

    def add(self, data_dict):
        with self.env.begin(write=True) as txn:
            for key, value in data_dict.items():
                txn.put(
                    key = key.encode(),
                    value = pickle.dumps(value)
                )

    def add_one(self, key, value):
        with self.env.begin(write=True) as txn:
            txn.put(
                key = key.encode(),
                value = pickle.dumps(value)
            )

    def close(self):
        self.env.close()

    def __getitem__(self, key):
        if isinstance(key, int):
            key = str(key)
        with self.env.begin() as txn:
            value = txn.get(key.encode())
        if value is None:
            return None
        else:
            return pickle.loads(value)
    
    def __len__(self):
        with self.env.begin() as txn:
            return txn.stat()['entries']
        
    def get_all_keys(self):
        with self.env.begin() as txn:
            return [k.decode() for k in txn.cursor().iternext(values=False)]


# @register_database('csd')  # pbdock use same logic as geom
# @register_database('pbdock')  # pbdock use same logic as geom
# @register_database('geom')
class SingleDatabase(Dataset): # modify from Drug3DDataset. directly the keys
    def __init__(self, config, root):
        super().__init__()
        self.config = config
        lmdb_root = config['lmdb_root']
        self.lmdb_path = {key: os.path.join(root, lmdb_root, value) for key, value in config['lmdb_path'].items()}
        
        self.db_dict = None
        self._connect_db()
        
    def _connect_db(self):
        """
            Establish read-only database connection
        """
        self.db_dict = {}
        for lmdb_name, lmdb_path in self.lmdb_path.items():
            self.db_dict[lmdb_name] = LMDBDatabase(lmdb_path)

    def _close_db(self):
        for db in self.db_dict.items():
            db.close()
        self.db_dict = None

    def __len__(self):
        raise NotImplementedError('Please implement __len__ method')

    def __getitem__(self, key):
        # if self.db_dict is None:
        #     self._connect_db()
        
        key_split = key.split(';')
        for i, key_this_db in enumerate(key_split):
            key_list = key_this_db.split('/')
            if len(key_list) == 2:
                lmdb_name, data_id = key_list
                fetch = self.db_dict[lmdb_name][data_id]
            elif len(key_list) == 3:
                lmdb_name, data_id, index_sep = key_list
                fetch = self.db_dict[lmdb_name]['/'.join([data_id, index_sep])]
                # fetch = self.db_dict[lmdb_name][data_id][int(index_sep)]
            # initial fetch
            if i == 0:
                # assert lmdb_name == 'mols' or lmdb_name.startswith('pocmol'),\
                #     'You must fetch mols or pocmol lmdb as the first lmdb since it contains torch.Data'
                data = fetch
            else:
                data.update(fetch)
        
        return data

