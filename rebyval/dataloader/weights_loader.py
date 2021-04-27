import tensorflow as tf
from rebyval.dataloader.base_dataloader import BaseDataLoader


class DnnWeightsLoader(BaseDataLoader):
    def __init__(self, dataloader_args):
        super(DnnWeightsLoader, self).__init__(dataloader_args=dataloader_args)
