import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
    
def load_tensors_from_dir(path, valid_extensions=None):
    tensors = []
    for file in os.listdir(path):
        if valid_extensions is not None:
            name, ext = os.path.splitext(file)
            if ext[1:] not in valid_extensions:
                continue
        tensor = torch.load(os.path.join(path, file))
        tensors.append(tensor)

    try:
        output = torch.stack(tensors)
    except Exception as e:
        raise "All tensors must have the same size"

    return output

def load_images_from_dir(path, valid_extensions=None):
    to_tensor = ToTensor()
    images = []
    for file in os.listdir(path):
        if valid_extensions is not None:
            name, ext = os.path.splitext(file)
            if ext[1:] not in valid_extensions:
                continue
        img = Image.open(os.path.join(path, file))
        images.append(to_tensor(img))

    try:
        output = torch.stack(images)
    except Exception as e:
        raise "All images must have the same size"
    
    return output