# Import libraries
import numpy as np
import os
import torch
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Transformations on images
transform = transforms.Compose([           
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Implementation of the train class 
class TrainDataset(Dataset):
    def __init__(self, root_dir, transform, img_per_place=4):
        """
        Initializes the TrainDataset.

        Args:
            root_dir (str): Root directory containing city directories with images.
            transform (callable): Transformations to be applied to the images.
            img_per_place (int): Number of images to keep for the same place ID. Default is 4.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_per_place= img_per_place  # number of images to keep for the same id
        self.place_paths = defaultdict(list)

        # Iteration loop trough all the directories of cities
        for city_dir in os.listdir(root_dir):
            city_path = os.path.join(root_dir, city_dir)

            # Check if it's a directory
            if os.path.isdir(city_path):
                # Iteration loop trough all the images of a city
                for filename in os.listdir(city_path):
                    img_path = os.path.join(city_path, filename)
                    place_id = img_path.split("@")[-2]
                    self.place_paths[place_id].append(img_path)
                    
        for place_id in list(self.place_paths.keys()):
            paths_place_id = self.place_paths[place_id]
            # Keep only the places that have at least a minimum of 4 images per id
            if len(paths_place_id) < 4: 
                del self.place_paths[place_id]
        self.places_ids = sorted(list(self.place_paths.keys()))
                 
                    
    def __getitem__(self, idx):
        """
        Retrieves a set of images and their corresponding place ID.

        Args:
            idx (int): Index of the place ID to retrieve.

        Returns:
            tuple: A tuple containing:
                - A tensor of stacked images.
                - A tensor of repeated indices corresponding to the place ID.
                - The place ID.
        """
        place_id = self.places_ids[idx]
        paths_place_id = self.place_paths[place_id]
        # Keep 4 random paths for each id
        chosen_paths = np.random.choice(paths_place_id, self.img_per_place)         
        images = [Image.open(path).convert('RGB') for path in chosen_paths]
        images = [self.transform(img) for img in images]
        return torch.stack(images), torch.tensor(idx).repeat(self.img_per_place), place_id
    
    def __len__(self):
        """
        Returns the number of unique place IDs in the dataset.
        """
        return len(self.places_ids)
    