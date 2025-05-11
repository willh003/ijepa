import argparse
import copy
import multiprocessing
import re
import warnings
import os
import time
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from earthnet_models_pytorch.utils import str2bool
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from earthnet_models_pytorch.data.en21x_data import EarthNet2021XDataset
from causal_earth.data.earthnet import EarthNetCollator
from ijepa.src.masks.multiblock import MaskCollator
from logging import getLogger
warnings.simplefilter(action="ignore", category=FutureWarning)

logger = getLogger()

class IjepaEarthNetDataset(EarthNet2021XDataset):
    """Dataset class for I-JEPA training on EarthNet data.
    
    This dataset:
    1. Inherits from EarthNet2021XDataset to get the base data loading functionality
    2. Uses EarthNetCollator to transform the data into RGB images
    3. Returns images and targets in the format expected by I-JEPA training
    """
    
    def __init__(
        self,
        folder: Union[Path, str],
        fp16=False,
        s2_bands=["ndvi", "B02", "B03", "B04", "B8A"],
        eobs_vars=["fg", "hu", "pp", "qq", "rr", "tg", "tn", "tx"],
        eobs_agg=["mean", "min", "max"],
        static_vars=["nasa_dem", "alos_dem", "cop_dem", "esawc_lc", "geom_cls"],
        start_month_extreme=None,
        dl_cloudmask=False,
        allow_fastaccess=False,
        transform=None
    ):
        """Initialize the dataset.
        
        Args:
            folder: Path to the dataset folder
            fp16: Whether to use float16 precision
            s2_bands: List of Sentinel-2 bands to use
            eobs_vars: List of E-OBS variables to use
            eobs_agg: List of E-OBS aggregations to use
            static_vars: List of static variables to use
            start_month_extreme: Optional start month for extreme events
            dl_cloudmask: Whether to use deep learning cloud mask
            allow_fastaccess: Whether to allow fast access to data
            transform: Optional transform to apply to the images
        """
        super().__init__(
            folder=folder,
            fp16=fp16,
            s2_bands=s2_bands,
            eobs_vars=eobs_vars,
            eobs_agg=eobs_agg,
            static_vars=static_vars,
            start_month_extreme=start_month_extreme,
            dl_cloudmask=dl_cloudmask,
            allow_fastaccess=allow_fastaccess
        )
        self.collator = EarthNetCollator(transform=transform)
        
    def __getitem__(self, idx: int) -> tuple:
        """Get a data sample.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            tuple: (image, target) where:
                - image: Transformed RGB image tensor
                - target: Land cover class as target
        """
        # Get the raw data from parent class
        data = super().__getitem__(idx)
        
        # Use collator to transform the data into RGB images
        images = self.collator([data])
        
        return images.squeeze(0)

def make_earthnet_ijepa(
    transform,
    batch_size,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    fp16=False,
    s2_bands=["ndvi", "B02", "B03", "B04", "B8A"],
    eobs_vars=["fg", "hu", "pp", "qq", "rr", "tg", "tn", "tx"],
    eobs_agg=["mean", "min", "max"],
    static_vars=["nasa_dem", "alos_dem", "cop_dem", "esawc_lc", "geom_cls"],
    start_month_extreme=None,
    dl_cloudmask=False,
    allow_fastaccess=False,
    drop_last=True,
    collator=None
):
    """Create EarthNet dataset and dataloader for I-JEPA training.
    
    Args:
        transform: Transform to apply to images
        batch_size: Batch size for dataloader
        pin_mem: Whether to pin memory in dataloader
        num_workers: Number of worker processes for dataloader
        world_size: Number of distributed processes
        rank: Rank of current process
        root_path: Path to dataset root
        fp16: Whether to use float16 precision
        s2_bands: List of Sentinel-2 bands to use
        eobs_vars: List of E-OBS variables to use
        eobs_agg: List of E-OBS aggregations to use
        static_vars: List of static variables to use
        start_month_extreme: Optional start month for extreme events
        dl_cloudmask: Whether to use deep learning cloud mask
        allow_fastaccess: Whether to allow fast access to data
        drop_last: Whether to drop last incomplete batch
        collator: Optional MaskCollator for I-JEPA training
        
    Returns:
        tuple: (dataset, dataloader, sampler)
    """
    # Create dataset
    dataset = IjepaEarthNetDataset(
        folder=root_path,
        fp16=fp16,
        s2_bands=s2_bands,
        eobs_vars=eobs_vars,
        eobs_agg=eobs_agg,
        static_vars=static_vars,
        start_month_extreme=start_month_extreme,
        dl_cloudmask=dl_cloudmask,
        allow_fastaccess=allow_fastaccess,
        transform=transform
    )
    logger.info('EarthNet dataset created')
    
    # Create distributed sampler
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    # Create dataloader with mask collator if provided
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    logger.info('EarthNet unsupervised data loader created')
    
    return dataset, data_loader, dist_sampler
