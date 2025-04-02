import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from torchvision import transforms
from PIL import Image
import numpy as np
import os

if __name__ == "__main__":
    # choose device and load MTCNN
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.5, 0.5, 0.5], margin=20)
    
    # load image, pixel range: 0-255
    image = cv2.imread("./dataset/test_image/Aaron_Guiel.jpg")
    
    # print(image.shape)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    
    # detect all face and crop, pixel range from -1 to 1, size [C, W, H]
    image_cropped = mtcnn(image)[0].numpy()
    
    # denormalize image to 0-255
    image_cropped = (image_cropped + 1) * 127.5
    image_cropped = np.clip(image_cropped, 0, 255).astype(np.uint8)
    
    # change image shape from [C, W, H] to [W, H, C]
    image_cropped = np.transpose(image_cropped, (1, 2, 0))
    
    # display cropped image
    cv2.imshow("Image", image_cropped)
    cv2.waitKey(0)