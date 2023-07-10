import os
from glob import glob
import natsort
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch

from lib_utils import rgb2mask

class CoreDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path_images = natsort.natsorted(glob(os.path.join(path, 'images', '*.tif')))
        self.path_masks = natsort.natsorted(glob(os.path.join(path, 'masks', '*.tif')))
        self.transform = transform
        
    def __len__(self):
        return len(self.path_images)
    
    def __getitem__(self, idx):
        image = Image.open(self.path_images[idx])
        mask = Image.open(self.path_masks[idx])
        
        # Vérifier le nombre de canaux et le mode de l'image
        if image.mode != 'RGBA':
            # Si le mode n'est pas RGBA, convertir l'image en mode RGBA
            image = image.convert('RGBA')
        
        # Extraire les trois premiers canaux (red, green, blue)
        image = image.split()[:3]
        image = Image.merge('RGB', image)
        
        image = np.array(image)
        mask = np.array(mask)
        annotation_ = np.zeros((8, mask.shape[0], mask.shape[1]))

        for c in range(1,8):
            annotation_[c]= (mask == c)  # Créer un masque pour chaque classe
            
            if (mask == 0).all(): 
                pass#print("all black") 
            #annotation_[c] = torch.from_numpy(mask.astype(np.float32))   # Assigner le masque à chaque canal correspondant
            

        sample = {'image': image, 'mask': annotation_}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


    
# add image normalization transform at some point
   
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        image, mask = sample['image'], sample['mask']  
        # standard scaling would be probably better then dividing by 255 (subtract mean and divide by std of the dataset)
        image = np.array(image)/255
        # convert colors to "flat" labels
        #mask = rgb2mask(np.array(mask))
        sample = {'image': torch.from_numpy(image).permute(2,0,1).float(),
                  'mask': torch.from_numpy(mask).float(), 
                 }
        
        return sample
    
def make_datasets(path, val_ratio):
    dataset = CoreDataset(path, transform = transforms.Compose([ToTensor()]))
    val_len = int(val_ratio*len(dataset))
    lengths = [len(dataset)-val_len, val_len]
    train_dataset, val_dataset = random_split(dataset, lengths)
    
    return train_dataset, val_dataset


def make_dataloaders(path, val_ratio, params):
    train_dataset, val_dataset = make_datasets(path, val_ratio)
    train_loader = DataLoader(train_dataset, drop_last=True, **params)
    val_loader = DataLoader(val_dataset, drop_last=True, **params)
    
    return train_loader, val_loader