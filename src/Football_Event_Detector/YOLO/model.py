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
                yolo_dict = info['Architecture YOLOV1']
            except yaml.YAMLError as err:
                print(err)
                return
        
        self.in_ch = kwargs.get('in_channels', 3)
        
        #conv && maxpool layers 
        self.layers : list = []
        for layer in yolo_dict:
            if 'layer' in layer:
                match layer['layer']:
                    case 'conv':
                        self.layers += [CNN_LAYER(self.in_ch, layer['filters'], kernel_size=layer['kernel_size'],
                                                     stride=layer['stride'],padding=layer['padding'])]
                        self.in_ch = self.in_ch, layer['filters']
                    case 'maxpool':
                        self.layers += [nn.MaxPool2d(kernel_size=(layer['size'],layer['size']) 
                                                     ,stride=(layer['stride'],layer['stride']))]
            
            else:
                for _ in range(layer['repeats']):
                    for j in range(len(layer['block'])):
                        self.layers += [CNN_LAYER(self.in_ch, layer['block'][j]['filters'], kernel_size=layer['block'][j]['kernel_size'],
                                                     stride=layer['block'][j]['stride'],padding=layer['block'][j]['padding'])]

                        self.in_ch = self.in_ch, layer['filters']
                
        self.dn = nn.Sequential(*self.layers) 
        
        #hard coded for now, will change, simple calc
        con1 = nn.Linear(in_features=yolo_dict[-1]['filters']*7*7, out_features=4096)
        #each split and 20 classes and 2 bounding boxes per cell
        conn2 = nn.Linear(in_features=4096, out_features=7*7*(20+2*5))
        
        
        self.fcl(nn.Flatten(), con1, nn.Dropout(0.5), nn.LeakyReLU(0.1), conn2)
        
            
    
    
    def forward_pass(self, val: any)-> None:
        return self.fcl(torch.flatten(self.dn(val), start_dim=1))  

        
        
        
        

class CNN_LAYER(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CNN_LAYER, self).__init__()
        
        self.in_ch = args[0]
        self.out_ch = args[1]
        self.__conv_layer(**kwargs)
    
    
    def __conv_layer(self, **kwargs) -> None:
        self.conv_layer = nn.Conv2d(self.in_ch, self.out_ch, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(self.out_ch) # speed up training process 
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    
    def forward_pass(self, val: any)-> None:
        return self.leaky_relu(self.batch_norm(self.conv_layer(val)))
    
    
    
    

class Yolo_v1_loss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(Yolo_v1_loss, self).__init__()
        self.classes = kwargs.get('classes', 20)
        self.boxes = kwargs.get('boxes', 2)
        self.split = kwargs.get('split', 7)
        #based on paper, focus on points not on object
        self.no_obj = kwargs.get('no_obj_lambda', 0.5)
        self.co_ord = kwargs.get('coord_lambda', 5)
        
        self.mse = nn.MSELoss(reduction="sum")
        

        
    
    
        
        

