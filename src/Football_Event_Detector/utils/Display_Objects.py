import cv2
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from Football_Event_Detector.YOLO import yolo_v1, utility_functions
import time
from IPython.display import clear_output
from yolov5 import YOLOv5



class ShowPlayers:
    def __init__(self) -> None:
        self.model = 