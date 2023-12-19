import torch
import torch.nn as nn
import yaml
import pkg_resources

class Yolo(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)        
        with pkg_resources.resource_stream(__name__,'architecture.yml') as f:
            try:
                info = yaml.safe_load(f)
            except yaml.YAMLError as err:
                print(err)
        
        
        print(info)
        
        
        
        

class CNN_LAYER(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.in_ch = args[0]
        self.out_ch = args[1]
        self.__conv_layer(**kwargs)
    
    
    def __conv_layer(self, **kwargs) -> None:
        self.conv_layer = nn.Conv2d(self.in_ch, self.out_ch, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(self.out_ch) # speed up training process 
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    
    def forward_pass(self, val: any)-> None:
        return self.leaky_relu(self.batch_norm(self.conv_layer(val)))
    
    
    
    
    
    
    
        
        

