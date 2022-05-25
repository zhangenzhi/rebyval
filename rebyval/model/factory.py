from .dnn import DNN
from .cnn import CNN
from .resnet import ResNet50



class ModelFactory():
    def __init__(self) -> None:
        self.model_list = {'dnn':DNN, 'cnn':CNN, 'resnet':ResNet50}

    def __call__(self, model_args):
        return self.get_model(model_args=model_args)
    
    def get_model(self, model_args):
        model_cls = self.model_list[model_args['name']]
        model = model_cls(**model_args)
        return model
  

model_factory = ModelFactory()

