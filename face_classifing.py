import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from torchvision import transforms
from PIL import Image
import numpy as np
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    mtcnn = MTCNN(keep_all=True, device=device)
    model = InceptionResnetV1(pretrained="casia-webface", classify=True).to(device)
    model.eval()
    
    image = cv2.imread("./dataset/raw/lfw-deepfunneled/Aaron_Guiel/Aaron_Guiel_0001.jpg")
    image_cropped = mtcnn(image)[0]
    
    logits = model(image_cropped.unsqueeze(0).to(device))
    
    probs = torch.softmax(logits, dim=1)
    print(probs.max())
    