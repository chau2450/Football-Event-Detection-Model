import torch
import torch.nn as nn
import yaml
import pkg_resources



class Yolo(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.layers = [
            # First Block of Conv and MaxPool
            CNN_LAYER(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second Block of Conv and MaxPool
            CNN_LAYER(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Additional Conv Layers
            CNN_LAYER(192, 128, kernel_size=1),
            CNN_LAYER(128, 256, kernel_size=3, padding=1),
            CNN_LAYER(256, 256, kernel_size=1),
            CNN_LAYER(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Repeating Layers
            *[CNN_LAYER(512 if i % 2 == 0 else 256, 512 if i % 2 == 1 else 256, kernel_size=1 if i % 2 == 0 else 3, padding=0 if i % 2 == 0 else 1) for i in range(4)],  # Repeating 4 times
            CNN_LAYER(512, 512, kernel_size=1),
            CNN_LAYER(512, 1024, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # More Conv Layers
            CNN_LAYER(1024, 512, kernel_size=1),
            CNN_LAYER(512, 1024, kernel_size=3, padding=1),
            CNN_LAYER(1024, 512, kernel_size=1),
            CNN_LAYER(512, 1024, kernel_size=3, padding=1),
            CNN_LAYER(1024, 1024, kernel_size=3, padding=1),
            CNN_LAYER(1024, 1024, kernel_size=3, stride=2, padding=1),
            CNN_LAYER(1024, 1024, kernel_size=3, padding=1),
            CNN_LAYER(1024, 1024, kernel_size=3, padding=1)
        ]
        
        self.dn = nn.Sequential(*self.layers)

        # Fully connected layers
        self.fcl = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 7 * 7 * (20 + 2 * 5))
        )

    def forward(self, x):
        x = self.dn(x)
        return self.fcl(x)


class CNN_LAYER(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_ch, out_ch, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.batch_norm(self.conv_layer(x)))




















# class Yolo(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         """_summary_
#         """
#         super().__init__(*args, **kwargs)        
#         with pkg_resources.resource_stream(__name__,'architecture.yml') as f:
#             try:
#                 info = yaml.safe_load(f)
#                 yolo_dict = info['Architecture YOLOV1']
#             except yaml.YAMLError as err:
#                 print(err)
#                 return
        
#         self.in_ch = kwargs.get('in_channels', 3)
        
#         #conv && maxpool layers 
#         self.layers : list = []
#         for layer in yolo_dict:
#             if 'layer' in layer:
#                 match layer['layer']:
#                     case 'conv':
#                         self.layers += [CNN_LAYER(self.in_ch, layer['filters'], kernel_size=layer['kernel_size'],
#                                                      stride=layer['stride'],padding=layer['padding'])]
#                         self.in_ch = layer['filters']
#                     case 'maxpool':
#                         self.layers += [nn.MaxPool2d(kernel_size=(layer['size'],layer['size']) 
#                                                      ,stride=(layer['stride'],layer['stride']))]
            
#             else:
#                 for _ in range(layer['repeats']):
#                     for j in range(len(layer['block'])):
#                         self.layers += [CNN_LAYER(self.in_ch, layer['block'][j]['filters'], kernel_size=layer['block'][j]['kernel_size'],
#                                                      stride=layer['block'][j]['stride'],padding=layer['block'][j]['padding'])]

#                         self.in_ch = layer['block'][j]['filters']
                
#         self.dn = nn.Sequential(*self.layers) 
        
#         #hard coded for now, will change, simple calc
#         con1 = nn.Linear(in_features=1024*7*7, out_features=4096)
#         #each split and 20 classes and 2 bounding boxes per cell
#         conn2 = nn.Linear(in_features=4096, out_features=7*7*(20+2*5))
        
        
#         self.fcl= nn.Sequential(nn.Flatten(), con1, nn.Dropout(0.5), nn.LeakyReLU(0.1), conn2)
        
            
#         print(self.layers)
    
#     def forward(self, val: any)-> None:
#         return self.fcl(torch.flatten(self.dn(val), start_dim=1))  


# class CNN_LAYER(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super(CNN_LAYER, self).__init__()
        
#         self.in_ch = args[0]
#         self.out_ch = args[1]
#         self.__conv_layer(**kwargs)
    
    
#     def __conv_layer(self, **kwargs) -> None:
#         self.conv_layer = nn.Conv2d(self.in_ch, self.out_ch, bias=False, **kwargs)
#         self.batch_norm = nn.BatchNorm2d(self.out_ch) 
#         self.leaky_relu = nn.LeakyReLU(0.1)
    
    
#     def forward(self, val: any)-> None:
#         return self.leaky_relu(self.batch_norm(self.conv_layer(val)))
    
    
    
    


        
        

        
    
    
        
        

