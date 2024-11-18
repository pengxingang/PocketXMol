import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import shutil
import sys
import argparse
import gc

from typing import Any, Callable, Optional, Union
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
import torch
# from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
torch.set_float32_matmul_precision('medium')

sys.path.append('.')
from models.maskfill import PMAsymDenoiser
from models.loss import get_loss_func
from utils.dataset import ForeverTaskDataset
from utils.transforms import FeaturizeMol, Compose, get_transforms, FeaturizePocket
from utils.misc import *
from utils.train import get_optimizer, get_scheduler, GradualWarmupScheduler
from utils.sample_noise import get_sample_noiser


def copy_py_files(src_dir, dst_dir, base=False):
    os.makedirs(dst_dir, exist_ok=True)
    for item in os.listdir(src_dir):
        # Get the absolute path of the item
        item_path = os.path.join(src_dir, item)
        if os.path.isdir(item_path):
            if (not base) or (item in ['scripts', 'models', 'notebooks', 'utils', 'process', 'evaluate']):
                # If the item is a directory, recursively call the function on it
                copy_py_files(item_path, os.path.join(dst_dir, item))
        elif (item.endswith('.py') or item.endswith('.sh') or item.endswith('.ipynb')):
            # If the item is a file and ends with .py, copy it to the destination directory
            shutil.copy(item_path, dst_dir)


class DataModule(pl.LightningDataModule):
    def __init__(self, config, ):
        super().__init__()
        self.config = config
        # self.multi_node = args.multi_node
        
    def get_featurizers(self):
        featurizer = FeaturizeMol(self.config.transforms.featurizer)
        if 'featurizer_pocket' in self.config.transforms:
            feat_pocket = FeaturizePocket(self.config.transforms.featurizer_pocket)
            return [feat_pocket, featurizer]  # pocket first because mol need to substract pocket center
        else:
            return [featurizer]

    def get_in_dims(self, featurizers=None):
        if featurizers is None:
            featurizers = self.get_featurizers()
        num_node_types = featurizers[-1].num_node_types
        num_edge_types = featurizers[-1].num_edge_types
        in_dims = {
            'num_node_types': num_node_types,
            'num_edge_types': num_edge_types,
        }
        if len(featurizers) == 2:
            in_dims.update({
                'pocket_in_dim': featurizers[0].feature_dim,
            })
        return in_dims
        
    def setup(self, stage=None):
        
        # # Transforms
        featurizers = self.get_featurizers()
        in_dims = self.get_in_dims(featurizers)
        task_trans = get_transforms(self.config.transforms.task, mode='train',
                                    num_node_types=in_dims['num_node_types'],)
        noiser = get_sample_noiser(self.config.noise, in_dims['num_node_types'], in_dims['num_edge_types'],
                                   mode='train')
        transform_list = featurizers + [task_trans, noiser]
        if 'cut_peptide' in self.config.transforms:
            transform_list = [get_transforms(self.config.transforms.cut_peptide)] + transform_list
        self.transforms = Compose(transform_list)
        follow_batch = sum([getattr(t, 'follow_batch', []) for t in self.transforms.transforms], [])
        exclude_keys = sum([getattr(t, 'exclude_keys', []) for t in self.transforms.transforms], [])
        # self.num_node_types = in_dims['num_node_types']
        # self.num_edge_types = in_dims['num_edge_types']

        # # Datasets and sampler
        data_cfg = self.config.data
        num_samplers_args = {'num_workers': self.config.train.num_workers,
                             'global_rank': self.trainer.global_rank,
                             'world_size': self.trainer.world_size}
        train_set = ForeverTaskDataset(data_cfg.dataset, data_cfg.task_db_weights,'train',
                                       transforms=self.transforms, shuffle=True, **num_samplers_args)
        if num_samplers_args['world_size'] > 100:
            divider = 4
        else:
            divider = 1
        num_samplers_args['num_workers'] = self.config.train.num_workers//divider
        val_set = ForeverTaskDataset(data_cfg.dataset, data_cfg.task_db_weights, 'val',
                                     transforms=self.transforms, shuffle=False, **num_samplers_args)

        # # Dataloaders
        train_cfg = self.config.train
        self.train_loader = DataLoader(train_set, batch_size=train_cfg.batch_size if not is_vscode else 40,
                                       num_workers=train_cfg.num_workers, pin_memory=train_cfg.pin_memory,
                                       follow_batch=follow_batch, exclude_keys=exclude_keys,
                                       persistent_workers=train_cfg.persistent_workers,
        )
        self.val_loader = DataLoader(val_set, batch_size=train_cfg.batch_size if not is_vscode else 40,
                                     num_workers=train_cfg.num_workers//divider, pin_memory=train_cfg.pin_memory,
                                     follow_batch=follow_batch, exclude_keys=exclude_keys,
                                     persistent_workers=train_cfg.persistent_workers,
        )
    def train_dataloader(self):
        return self.train_loader
    def val_dataloader(self):
        return self.val_loader


class ModelLightning(pl.LightningModule):
    def __init__(self, config, args, num_node_types, num_edge_types, **kwargs):
        super(ModelLightning, self).__init__()
        self.config = config
        self.save_hyperparameters()
        self.num_gpus = args.num_gpus
        self.multi_node = args.multi_node
        self.sync_dist = (self.num_gpus>1) or self.multi_node

        # Model
        if self.config.model.name == 'pm_asym_denoiser':
            self.model = PMAsymDenoiser(config=self.config.model,
                                  num_node_types=num_node_types,
                                  num_edge_types=num_edge_types, **kwargs)
        
        if getattr(self.config.model, 'pretrained', ''):
            ckpt = torch.load(self.config.model.pretrained, map_location='cpu')
            self.model.load_state_dict({k[6:]:value for k, value in ckpt['state_dict'].items()
                                        if k.startswith('model.')})
            print('Load pretrained model from', self.config.model.pretrained)
        

        self.loss_func = get_loss_func(self.config.loss)
        # self.skip = False
        # self.automatic_optimization = False
        # self.gradient_clip_val = getattr(self.config.train, 'gradient_clip_val', 0)
        # self.gradient_clip_algorithm = getattr(self.config.train, 'gradient_clip_algorithm', 'norm')

    def forward(self, batch):
        return self.model(batch)
    
    def reduce_batch(self, batch):
        # free memory
        for p in self.model.parameters():
            if p.grad is not None:
                del p.grad  # free some memory
        # gc.collect()
        torch.cuda.empty_cache()

        # drop last 10 percent
        new_bs = int(len(batch) * 0.5)
        print(f"\nOut of memory error occurred in step. Reduce bs {len(batch)} to {new_bs}")
        device = batch.batch.device
        # follow_batch = [k.replace('_batch','') for k in batch.keys if k.endswith('_batch')]
        follow_batch = [k.replace('_batch','') for k in batch.keys() if k.endswith('_batch')]
        # batch_cpu = batch.detach().cpu()
        batch_cpu = batch.cpu()
        del batch
        # gc.collect()
        torch.cuda.empty_cache()
        data_list = batch_cpu.to_data_list()
        del data_list[new_bs:]
        batch = Batch.from_data_list(data_list[:new_bs], follow_batch=follow_batch).to(device)
        # size_list = [data.num_nodes for data in data_list]  # choose large first
        # idx_sort = np.argsort(size_list)[::-1]
        # batch = Batch.from_data_list([data_list[i] for i in idx_sort[:new_bs]],
        #                              follow_batch=follow_batch).to(device)
        return batch

    def training_step2(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        while True:
            try:
                opt.zero_grad()
                # forward
                outputs = self.model(batch)
                loss_dict = self.loss_func(batch, outputs)
                # bachward
                self.manual_backward(loss_dict['loss'])
                if self.gradient_clip_val > 0:
                    self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm=self.gradient_clip_algorithm)
                opt.step()
                break
            except Exception as e:
                if isinstance(e, RuntimeError) and "out of memory" in str(e):
                    opt.zero_grad()
                    # del outputs
                    try:
                        del outputs
                    except Exception as e:
                        print(e)
                        pass
                    gc.collect()
                    torch.cuda.empty_cache()
                    # del loss_dict
                    try:
                        del loss_dict
                    except Exception as e:
                        print(e)
                        pass
                    gc.collect()
                    torch.cuda.empty_cache()
                    if len(batch) >= 4:
                        batch = self.reduce_batch(batch)
                    else:
                        return None
                else:
                    raise e

        sch.step()
        loss = loss_dict['loss']
        loss_dict = {'train/'+k: v for k, v in loss_dict.items()}
        self.log_dict({k:v for k,v in loss_dict.items() if '_fixed/' not in k}, batch_size=batch.num_graphs,
                      sync_dist=self.sync_dist, prog_bar=True, logger=True)
        self.log_dict({k:v for k,v in loss_dict.items() if '_fixed/' in k}, batch_size=batch.num_graphs,
                      sync_dist=self.sync_dist, prog_bar=False, logger=True)
        # self.log('train_loss', loss, sync_dist=(self.num_gpus>1))
        # self.log('train/grad', orig_grad_norm)
        self.log('train/lr', opt.param_groups[0]['lr'], sync_dist=self.sync_dist)
        norms = grad_norm(self.model, norm_type=2)
        if norms:
            max_norms = max(norms.values())
            self.log_dict({'train/max_norm': max_norms}, sync_dist=self.sync_dist)
        return loss

    def on_before_optimizer_step(self, optimizer):
        # warmup lr increase
        if self.global_step < self.warmup_step:
            if self.global_step == 0:
                self.base_lr_list = []
                for param_group in self.optimizers().param_groups:
                    self.base_lr_list.append(param_group['lr'])
            ratio = float((self.global_step+1) / self.warmup_step)
            for i, param_group in enumerate(self.optimizers().param_groups):
                    param_group['lr'] = self.base_lr_list[i] * ratio
        
        
        self.log('train/lr', optimizer.param_groups[0]['lr'], sync_dist=self.sync_dist, prog_bar=True)
        norms = grad_norm(self.model, norm_type=2)
        if norms:
            max_norms = max(norms.values()).to(self.device)
            self.log_dict({'train/max_norm': max_norms}, sync_dist=self.sync_dist)

    # def on_after_backward(self) -> None:
    #     for name, param in self.model.named_parameters():
    #         if param.grad.isinf().any() or param.grad.isnan().any():
    #             print(f"Inf or NaN in {name}")
    #     return super().on_after_backward()

    def training_step(self, batch, batch_idx):

        while True:
            try:
                outputs = self.model(batch)
                loss_dict = self.loss_func(batch, outputs)
                # print('\n', len(batch.node_type_batch), len(batch.pocket_pos_batch))
                break
            except Exception as e:
                if isinstance(e, RuntimeError) and "out of memory" in str(e):
                    # zhale = 1.
                    print('\nOOM', len(batch.node_type_batch), len(batch.pocket_pos_batch))
                    # if len(batch) >= 4:
                    batch = self.reduce_batch(batch)
                    # else:
                    #     return None
                else:
                    raise e

        if 'loss' in loss_dict:
            loss = loss_dict['loss']
        else:
            loss = loss_dict['mixed/total']

        loss_dict = {'train_'+k: v for k, v in loss_dict.items()}
        # if self.global_step != 73:
        # if True:
        self.log_dict({k.replace('train_mixed', 'train'):v for k, v in loss_dict.items() if 'mixed/' in k}, batch_size=batch.num_graphs,
                    sync_dist=self.sync_dist, prog_bar=True, logger=True)
        self.log_dict({k:v for k, v in loss_dict.items() if 'mixed/' not in k}, batch_size=batch.num_graphs,
                    sync_dist=self.sync_dist, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        while True:
            try:
                outputs = self.model(batch)
                loss_dict = self.loss_func(batch, outputs)
                break
            except Exception as e:
                if isinstance(e, RuntimeError) and "out of memory" in str(e):
                    # try:
                    #     del outputs
                    #     del loss_dict
                    # except:
                    #     pass
                    # gc.collect()
                    # torch.cuda.empty_cache()
                    batch = self.reduce_batch(batch)
                else:
                    raise e
        
        loss_dict = {'val_'+k: v for k, v in loss_dict.items()}
        self.log_dict({k.replace('val_mixed', 'val'):v for k,v in loss_dict.items() if ('mixed/' in k) and ('fixed_' not in k)}, batch_size=batch.num_graphs,
                      sync_dist=self.sync_dist, prog_bar=True, logger=True)
        self.log_dict({k.replace('val_mixed', 'val'):v for k,v in loss_dict.items() if ('mixed/' in k) and ('fixed_' in k)}, batch_size=batch.num_graphs,
                      sync_dist=self.sync_dist, prog_bar=False, logger=True)
        self.log_dict({k:v for k,v in loss_dict.items() if ('mixed/' not in k)}, batch_size=batch.num_graphs,
                      sync_dist=self.sync_dist, prog_bar=False, logger=True)

        self.log('val/loss', loss_dict['val_mixed/total'], batch_size=batch.num_graphs,
                      sync_dist=self.sync_dist, prog_bar=False, logger=True)
        return loss_dict

    def configure_optimizers(self):
        # Optimizer and scheduler
        optimizer = get_optimizer(self.config.train.optimizer, self.model)
        scheduler_config = self.config.train.scheduler
        scheduler = get_scheduler(scheduler_config.instance, optimizer)
        self.warmup_step = getattr(scheduler_config, "warmup_step", 0)
        if self.warmup_step > 0:
            print('Warmup step is', self.warmup_step)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                **scheduler_config.params,
            },
        }


    def optimizer_step2(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        try:
            return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        except Exception as e:
            if isinstance(e, RuntimeError) and "out of memory" in str(e):
                print(f"\nOut of memory error occurred in optimizer_step. Skip.")
                self.skip = True
                torch.cuda.empty_cache()
                super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
            else:
                raise e


is_vscode = False
if os.environ.get("TERM_PROGRAM") == "vscode":
    is_vscode = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
        default='configs/train/train_pxm.yml')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--multi_node', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='GPU device id. Only for single GPU training.')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--logdir', type=str, default='lightning_logs_tasked')
    parser.add_argument('--profile', type=bool, default=False)
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()
    # log dir
    if is_vscode:
        args.logdir = os.path.join('vscode', args.logdir)
    dir_names = os.path.dirname(args.config).split('/')
    is_train = dir_names.index('train')
    names = dir_names[is_train+1:]
    args.logdir = '/'.join([args.logdir] + names)

    # Load configs
    config = make_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # data and model
    dm = DataModule(config)
    in_dims = dm.get_in_dims()
    model = ModelLightning(config, args, **in_dims)

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        filename='{step}',
        monitor='val/loss',
        mode='min',
        # save_top_k=3,
        save_top_k=-1,
        every_n_train_steps=config.train.ckpt_every_n_steps,
        save_last=True,
        verbose=True,
        # every_n_epochs=config.train.ckpt_every_n_epochs,
    )
    logger = TensorBoardLogger(
        save_dir=args.logdir,
        name=config_name,
        version=args.tag if args.tag else None,
    )

    if not args.multi_node:
        devices = args.num_gpus if args.num_gpus > 1 else [args.device]
        num_nodes = args.num_nodes
    else:
        devices = 1
        num_nodes = int(os.environ.get("NUM_NODES", 1))
    # print(args.num_nodes, args.num_gpus, devices)
    trainer = pl.Trainer(
        devices=devices,
        num_nodes=num_nodes,
        max_epochs=1,
        max_steps=config.train.max_steps,
        callbacks=[checkpoint_callback],
        precision=config.train.precision,
        check_val_every_n_epoch=None,
        log_every_n_steps=config.train.val_check_interval,
        val_check_interval=config.train.val_check_interval,
        gradient_clip_val=getattr(config.train, 'gradient_clip_val', None),
        logger=logger,
        num_sanity_val_steps=0,
        profiler='simple' if args.profile else None,
        strategy='ddp',
        # strategy='ddp_find_unused_parameters_true',
        accelerator='gpu',
        # detect_anomaly=True,
        # detect_anomaly=True
        # limit_train_batches=1.0 if not args.profile else 100,
        # limit_val_batches=1.0 if not args.profile else 50,
    )
    
    # resume
    log_dir = trainer.logger.log_dir
    if args.resume:
        ckpt_path = os.path.join(os.path.dirname(log_dir),
                        args.resume, 'checkpoints/last.ckpt')
        print('Resume from', ckpt_path)
        config['resume'] = ckpt_path
    else:
        ckpt_path = None
    
    # save source code (only for rank 0 if use multiple GPUs)
    if (not args.multi_node and ((args.num_gpus == 1) or (trainer.global_rank == 0))) or \
        (args.multi_node and trainer.global_rank == 0):
        curr_dir = '.' # os.path.dirname(os.path.realpath(__file__))
        save_dir = os.path.join(trainer.logger.log_dir, "src")
        copy_py_files(curr_dir, save_dir, base=True)
        config_dir = os.path.join(log_dir, 'train_config')
        os.makedirs(config_dir, exist_ok=True)
        save_config(config, os.path.join(config_dir, os.path.basename(args.config)))
        # shutil.copyfile(args.config, os.path.join(config_dir, os.path.basename(args.config)))

    trainer.fit(model, dm, ckpt_path=ckpt_path)
    print('Training finished!')
