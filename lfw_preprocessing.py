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
mtcnn = MTCNN(keep_all=False, device=device, thresholds=[0.5, 0.5, 0.5])
model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
model.eval()

if __name__ == "__main__":
    names, embeddings = [], []
    dataset_path = "./dataset/raw/lfw-deepfunneled"
    
    # all person
    for name in tqdm(os.listdir(dataset_path)):
        # one person, n is total image of this person
        embeds = []
        for img in os.listdir(f"{dataset_path}/{name}"):
            img_file = os.path.join(f"{dataset_path}/{name}", img)
            image = Image.open(img_file).convert("RGB")
            # [1, 3, 160, 160] if keep_all=True, [3, 160, 160] if keep_all=False, pixel range: [-1, 1] (normalized), it could be None if no face detected
            face_detected = mtcnn(image)
            
            if face_detected is None:
                continue
            
            with torch.no_grad():
                feature_vector = model(face_detected.unsqueeze(0).to(device)) # [1, 512]
                # feature_vector = model(face_detected.to(device)) # [1, 512]
                embeds.append(feature_vector)
        
        if len(embeds) == 0:
            continue
        
        embeds = torch.cat(embeds, dim=0) # [n, 512]
        final_embedding = torch.mean(embeds, dim=0) # [512]
        embeddings.append(final_embedding)
        names.append(name)
    
    torch.save(embeddings, "face_embeddings_vggface2.pt")
    np.save("face_names_vggface2.npy", names)
    print(len(names))
    print(len(embeddings))