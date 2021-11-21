from config import TaskConfig
from data_utils.Augs import AugsCreation
from data_utils.Collator import Collator
from data_utils.SpeechCommandDataset import SpeechCommandDataset
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Tuple

TRAIN_IDX_PATH = "data_utils/train_val_split/train_indexes.pt"
VAL_IDX_PATH = "data_utils/train_val_split/val_indexes.pt"

def buildDataloaders()->Tuple[DataLoader, DataLoader]:
    dataset = SpeechCommandDataset(
        path2dir='speech_commands', keywords=TaskConfig.keyword
    )
    
    assert os.path.isfile(TRAIN_IDX_PATH), f"You should run `fix_train_val_split()` before buildDataloader()."
    assert os.path.isfile(VAL_IDX_PATH), f"You should run `fix_train_val_split()` before buildDataloader()."

    train_indexes = torch.load(TRAIN_IDX_PATH)
    val_indexes = torch.load(VAL_IDX_PATH)
    train_df = dataset.csv.iloc[train_indexes].reset_index(drop=True)
    val_df = dataset.csv.iloc[val_indexes].reset_index(drop=True)

    train_set = SpeechCommandDataset(csv=train_df, transform=AugsCreation())
    val_set = SpeechCommandDataset(csv=val_df)

    # We should provide to WeightedRandomSampler _weight for every sample_; by default it is 1/len(target)
    def get_sampler(target):
        class_sample_count = np.array(
            [len(np.where(target == t)[0]) for t in np.unique(target)])   # for every class count it's number of occ.
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.float()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return sampler
    train_sampler = get_sampler(train_set.csv['label'].values)


    # create two times because of Datasphere things
    # Here we are obliged to use shuffle=False because of our sampler with randomness inside.
    train_loader = DataLoader(train_set, batch_size=TaskConfig.batch_size,
                            shuffle=False, collate_fn=Collator(),
                            sampler=train_sampler,
                            num_workers=0, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=TaskConfig.batch_size,
                            shuffle=False, collate_fn=Collator(),
                            num_workers=0, pin_memory=True)

    next(iter(train_loader))
    next(iter(val_loader))

    train_loader = DataLoader(train_set, batch_size=TaskConfig.batch_size,
                            shuffle=False, collate_fn=Collator(),
                            sampler=train_sampler,
                            num_workers=2, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=TaskConfig.batch_size,
                            shuffle=False, collate_fn=Collator(),
                            num_workers=2, pin_memory=True)
    return train_loader, val_loader

def fix_train_val_split():
    dataset = SpeechCommandDataset(
        path2dir='speech_commands', keywords=TaskConfig.keyword
    )
    
    indexes = torch.randperm(len(dataset))
    train_indexes = indexes[:int(len(dataset) * 0.8)]
    val_indexes = indexes[int(len(dataset) * 0.8):]

    torch.save(train_indexes, TRAIN_IDX_PATH)
    torch.save(val_indexes, VAL_IDX_PATH)
