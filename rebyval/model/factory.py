from .dnn import DNN
from .cnn import CNN
from .resnet import ResNet50, ResNet56
from .vgg import VGG16, VGG11
from tensorflow.keras.applications.resnet50 import ResNet50 as kResNet50


class ModelFactory():
    def __init__(self) -> None:
        self.model_list = {'dnn':DNN, 'cnn':CNN, 
                           'resnet50':ResNet50,
                           'resnet56':ResNet56, 
                           'vgg16':VGG16,
                           'vgg11':VGG11}

    def __call__(self, model_args):
        return self.get_model(model_args=model_args)
    
    def get_model(self, model_args):
        model_cls = self.model_list[model_args['name']]
        model = model_cls(**model_args)
        return model
  

model_factory = ModelFactory()

