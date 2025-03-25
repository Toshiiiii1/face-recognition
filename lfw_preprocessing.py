import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from torchvision import transforms
from PIL import Image
import numpy as np
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained="casia-webface", classify=True).to(device)

if __name__ == "__main__":
    print("Start preprocessing...")