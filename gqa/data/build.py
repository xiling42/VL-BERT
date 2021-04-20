import torch.utils.data

from .datasets import *
from .datasets.gqa import transform, collate_data
from . import samplers
from .transforms.build import build_transforms
from .collate_batch import BatchCollator
import pprint
from torch.utils.data import DataLoader


DATASET_CATALOGS = {'gqa': GQADataset}




def build_dataset(dataset_name, *args, **kwargs):
    assert dataset_name in DATASET_CATALOGS, "dataset not in catalogs"
    return DATASET_CATALOGS[dataset_name](*args, **kwargs)


def make_data_sampler(dataset, shuffle, distributed, num_replicas, rank):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle, num_replicas=num_replicas, rank=rank)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(dataset, sampler, aspect_grouping, batch_size):
    if aspect_grouping:
        group_ids = dataset.group_ids
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, batch_size, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=False
        )
    return batch_sampler


def make_dataloader(cfg,  dataset=None, mode='train', distributed=False, num_replicas=None, rank=None,
                    expose_sampler=False):
    num_gpu = len(cfg.GPUS.split(','))
    batch_size = cfg.TRAIN.BATCH_IMAGES * num_gpu
    shuffle = cfg.TRAIN.SHUFFLE
    num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
    if mode == 'train':
        dataset_object = GQADataset(cfg.DATASET.DATASET_PATH, transform=transform)
    
        train_set = DataLoader(
            dataset_object, batch_size=batch_size, shuffle = True, num_workers=num_workers, collate_fn=collate_data
        )
    
        
        # dataset = iter(train_set)
        return train_set
    if mode == 'val':
   
        dataset_object = GQADataset(cfg.DATASET.DATASET_PATH, split = 'val', transform=transform)
    
        val_set = DataLoader(
            dataset_object, batch_size=batch_size, shuffle = False, num_workers=num_workers, collate_fn=collate_data
        )
    
        # dataset = iter(train_set)
        return val_set
