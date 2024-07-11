# Import libraries
import numpy as np
import os
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from torchvision import transforms


# Transformations on images
transform = transforms.Compose([           
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Implementation of the evaluation classes (validation,test)
class EvalDataset(Dataset):
    def __init__(self, root_dir,type_of_set,transform):
        """
        Initializes the EvalDataset.

        Args:
            root_dir (str): Root directory containing validation or test directories.
            type_of_set (str): Type of dataset, either 'val' for validation or 'test' for test.
            transform (callable): Transformations to be applied to the images.
        
        Raises:
            ValueError: If type_of_set is not 'val' or 'test'.
        """
        self.root_dir = root_dir
        self.type_of_set = type_of_set
        self.transform = transform

        if (type_of_set != 'val') and (type_of_set != 'test'):
            raise ValueError(f"Type of set not valid,try 'val' or 'test'")
        else:
            path_directory = os.path.join(root_dir,type_of_set)
            database_dir = os.path.join(path_directory,'database')
            queries_dir = os.path.join(path_directory, 'queries')

        self.database_paths = []
        for filename in os.listdir(database_dir):
            img_path = os.path.join(database_dir, filename)
            self.database_paths.append(img_path)
        self.queries_paths = []
        for filename in os.listdir(queries_dir):
            img_path = os.path.join(queries_dir, filename)
            self.queries_paths.append(img_path)

        self.database_coordinates = np.array \
            ([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        
        self.queries_coordinates = np.array\
            ([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)

        # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_coordinates)
        self.positives_per_query = knn.radius_neighbors(self.queries_coordinates,
                                                        radius=25,
                                                        return_distance=False)
        # Create a unique list to ease the __getitem__
        self.all_images_paths = [path for path in self.database_paths]
        self.all_images_paths += [path for path in self.queries_paths]

        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)


    def __getitem__(self, idx):
        """
        Retrieves an image and its index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: A tuple containing:
                - A transformed image tensor.
                - The index of the image.
        """
        image_path = self.all_images_paths[idx]
        image = self.transform(Image.open(image_path).convert('RGB'))
        return image, idx

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.all_images_paths)

    def get_positives(self):
        """
        Returns the list of positive neighbors for each query.
        """
        return self.positives_per_query