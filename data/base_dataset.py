from torch.utils.data import Dataset
from abc import abstractmethod

class BaseDataset(Dataset):
    
    options = None
    data_dir = ''

    patch_size = 0
        
    def __init__(self, options, data_dir):
        super().__init__()
        self.options = options
        self.data_dir = data_dir        

        self.patch_size = self.options.get("patch_size")

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass


    