import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.5, 0.5, 0.5])
model = InceptionResnetV1(pretrained="casia-webface", classify=False).to(device)
model.eval()

if __name__ == "__main__":
    # 1 person
    # embeddings = []
    # for img in os.listdir("./dataset/raw/lfw-deepfunneled/Aaron_Peirsol"):
    #     img_path = os.path.join("./dataset/raw/lfw-deepfunneled/Aaron_Peirsol", img)
    #     image = cv2.imread(img_path)
    #     image_cropped = mtcnn(image) # [3, 160, 160], pixel range: [-1, 1] (normalized)
        
    #     with torch.no_grad():
    #         feature_vector = model(image_cropped.to(device)) # [1, 512]
    #         embeddings.append(feature_vector)
            
    # # print(embeddings) [4, 1, 512]
    # embeddings = torch.cat(embeddings, dim=0) # [4, 512]
    # final_embedding = torch.mean(embeddings, dim=0) # [512]
    # print(final_embedding)
    
    names, embeddings = [], []
    dataset_path = "./dataset/raw/lfw-deepfunneled"
    
    # all person
    for name in tqdm(os.listdir(dataset_path)):
        # one person, n is total image of this person
        embeds = []
        for img in os.listdir(f"{dataset_path}/{name}"):
            img_file = os.path.join(f"{dataset_path}/{name}", img)
            image = cv2.imread(img_file)
            face_detected = mtcnn(image) # [1, 3, 160, 160], pixel range: [-1, 1] (normalized), it could be None if no face detected
            
            if face_detected is None:
                continue
            
            with torch.no_grad():
                feature_vector = model(face_detected.to(device)) # [1, 512]
                embeds.append(feature_vector)
        
        if len(embeds) == 0:
            continue
        
        embeds = torch.cat(embeds, dim=0) # [n, 512]
        final_embedding = torch.mean(embeds, dim=0) # [512]
        embeddings.append(final_embedding)
        names.append(name)
    
    np.save("face_embeddings.npy", {"names": names, "embeddings": embeddings})
    print(len(names))
    print(len(embeddings))
    # print(names[0])
    # print(embeddings[0])