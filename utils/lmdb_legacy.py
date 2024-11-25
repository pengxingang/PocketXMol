import lmdb
import pickle
from torch.utils.data import Dataset
import pandas as pd

class LMDBDatabase(Dataset):
    def __init__(self, db_path, map_size=10e9, readonly=True):
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
                meminit=False,
            )
        else:
            self.env = lmdb.open(
                self.db_path,
                map_size=self.map_size,
                subdir=False,
                readonly=False,
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

    # def modify(self, key, new_value):
    #     with self.env.begin(write=True) as txn:
    #         new_value = new_value.encode('utf-8')
    #         txn.put(key.encode('utf-8'), new_value)

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
        
        
class LMDBDatabaseWithDf(LMDBDatabase):
    def __init__(self, db_path, df_path=None, **kwargs):
        super().__init__(db_path, **kwargs)
        df_path = df_path if df_path is not None else db_path.replace('.lmdb', '.csv')
        
        self.df = pd.read_csv(df_path, index_col=0)
        self.df.index = self.df.index.astype(str)
        
    def __getitem__(self, key):
        data = super().__getitem__(key)
        return [data, self.df.loc[key].to_dict()]
        
    
        
    