import torch
from .preprocessing import HARdataset

def build_dataloader(root, params):
    """Build train and val dataloaders."""
    dataset = HARdataset(root)

    train_sampler, val_sampler = dataset.split_ind(val_split=params["split"],
                                                   shuffle=True)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=params["batch_size"],
                                               num_workers=params["workers"],
                                               pin_memory=True,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=params["batch_size"],
                                             num_workers=params["workers"],
                                             pin_memory=True,
                                             sampler=val_sampler)
    return train_loader, val_loader