
from .dataset_loader import Cifar10DataLoader, MinistDataLoader, ImageNetDataLoader, Cifar100DataLoader
from .wdnn_loader import DNNWeightsLoader, DNNSumReduce

class DatasetFactory():
    def __init__(self) -> None:
        self.dataset_list = {'cifar10':Cifar10DataLoader, 
                             'cifar100':Cifar100DataLoader, 
                             'mnist':MinistDataLoader, 
                             'imagenet': ImageNetDataLoader,
                             
                             'dnn_flatten':DNNWeightsLoader,
                             'dnn_sumreduce':DNNSumReduce
                            }
    
    def __call__(self, dataloader_args):
        dataset = self.get_dataset(dataloader_args)
        return dataset

    def get_dataset(self, dataloader_args):
        dataset_cls = self.dataset_list.get(dataloader_args['name'])
        return dataset_cls(dataloader_args)
        
dataset_factory = DatasetFactory()