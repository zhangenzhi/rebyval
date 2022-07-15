class BaseDataLoader:
    def __init__(self, dataloader_args):
        self.dataloader_args = dataloader_args
        self.dataspec = self._build_dataset_spec()

    def _build_dataset_spec(self):
        return None

    def process_example(self, example):
        pass

    def load_dataset(self, format=None):
        raise NotImplementedError("Must be implement in sub class")
    
    def to_devicebag(self):
        pass
    
class DeviceBag:
    def __init__(self) -> None:
        pass
    
    def to_iter(self):
        
        pass
    
        
