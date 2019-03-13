from torch.utils.data import Dataset
from abc import abstractmethod, ABC

class BaseDataset(Dataset):
    """Base dataset"""
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.data_dir = manager.get_data_dir()

        self.patch_size = manager.get_hyperparams().get("patch_size")

    