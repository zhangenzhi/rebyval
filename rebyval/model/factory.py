from cgi import test
from .dnn import DNN
from .cnn import CNN
from .resnet import ResNet50, ResNet56, nResNet56
from .vgg import VGG16, VGG11
from .resnet_test import ResNet56ForCIFAR10


class ModelFactory():
    def __init__(self) -> None:
        self.model_list = {'dnn':DNN, 'cnn':CNN, 
                           'resnet50':ResNet50,
                           'resnet56':ResNet56, 
                           'nresnet56':nResNet56, 
                           't-resnet56':ResNet56ForCIFAR10, 
                           'vgg16':VGG16,
                           'vgg11':VGG11}

    def __call__(self, model_args):
        return self.get_model(model_args=model_args)
    
    def get_model(self, model_args):
        model_cls = self.model_list[model_args['name']]
        if model_args['name'] == 't-resnet56':
            return model_cls(input_shape=(32, 32, 3), classes=10, weight_decay=1e-4)
        model = model_cls(**model_args)
        return model
  

model_factory = ModelFactory()

