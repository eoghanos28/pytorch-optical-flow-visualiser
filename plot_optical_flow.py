import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T

#initialise parameters for pyplot
plt.rcParams["savefig.bbox"] = "tight"


#Function for plotting image
def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            #print("OK")

    plt.tight_layout()
    plt.show()
    
    




import tempfile
from pathlib import Path
from urllib.request import urlretrieve


video_path = "oin.mp4"
#print("img read ok")

#Read the video located at video_path (oin.mpf in this case)
from torchvision.io import read_video
#Get frames from video path and make them tensors
frames, _, _ = read_video(str(video_path))


#Rotate the tensor (.permute)
frames = frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#Take frames 49 to 73 and 51 to 75
img1_batch = torch.stack([frames[50], frames[74]])
img2_batch = torch.stack([frames[51], frames[75]])
plot(img1_batch)


def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(520, 960)),
        ]
    )
    batch = transforms(batch)
    return 


#checks if you have cuda available on your graphics card which allows fast running of ML programs
device = "cuda" if torch.cuda.is_available() else "cpu"

#process the images with the device (cuda or cpu)
img1_batch = preprocess(img1_batch).to(device)
img2_batch = preprocess(img2_batch).to(device)

#print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")





#Import raft model (pre-trained opitcal flow model)
from torchvision.models.optical_flow import raft_large

model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()

list_of_flows = model(img1_batch.to(device), img2_batch.to(device))

predicted_flows = list_of_flows[-1]


#predicted_flows now contains the optical flow. torchvision has an inbuilt function to convert to rgb images 
from torchvision.utils import flow_to_image

flow_imgs = flow_to_image(predicted_flows)

#Plot the resulting image with matplotlib pyplot
img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
plot(grid)
