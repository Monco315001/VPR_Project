# Import libraries
import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch
from prettytable import PrettyTable
from visualization.Visualization import print_preds

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device


def recall(dataset, database_descriptors, queries_descriptors, k_values, print_predictions, number_of_images_per_epoch):
    """
    Calculate recall at different values of k using Faiss for optimized search.

    Args:
        dataset (object): Dataset object containing queries and true positives.
        database_descriptors (numpy.ndarray): Descriptors of the database images.
        queries_descriptors (numpy.ndarray): Descriptors of the query images.
        k_values (list): List of integers for k values to calculate recall.
        print_predictions (bool): Flag to print predictions with images if True.
        number_of_images_per_epoch (int): Maximum number of images to display per epoch.

    Returns:
        numpy.ndarray: Array of recall values at each k value.
    """
    # Use faiss to optimize the research
    faiss_index = faiss.IndexFlatL2(queries_descriptors.shape[1])
    faiss_index.add(database_descriptors)
    del database_descriptors
    
    # Perform nearest neighbor search
    _, predictions = faiss_index.search(queries_descriptors, max(k_values))
 
    positives_per_query = dataset.get_positives()
    recalls = np.zeros(len(k_values))
        
    # Calculate recall for each query
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            if np.any(np.in1d(pred[:n], positives_per_query[q_idx])):
                recalls[i:] += 1
                break
                
    recalls = recalls / dataset.queries_num * 100    
    
    if print_predictions == True:
        # For each query save 3 predictions
        print_preds(predictions[:, :3],dataset,number_of_images_per_epoch)
    
    # Display recall values in a table
    table = PrettyTable()
    table.field_names = ['K']+[str(k) for k in k_values]
    table.add_row(['Recall@K']+ [f'{values:.2f}' for values in recalls])
    print(table)
    
    return recalls


def evaluation_loop(dataset, model, dataloader, k_values,print_predictions, number_of_images_per_epoch = 5):
    """
    Evaluate the model's performance using recall at different values of k.

    Args:
        dataset (object): Dataset object containing queries and database images.
        model (nn.Module): Model to be evaluated.
        dataloader (DataLoader): DataLoader for the evaluation data.
        k_values (list): List of integers for k values to calculate recall.
        print_predictions (bool): Flag to print predictions with images if True.
        number_of_images_per_epoch (int, optional): Maximum number of images to display per epoch. Default is 5.

    Returns:
        numpy.ndarray: Array of recall values at each k value.
    """
    model.eval()
    recalls = np.zeros(len(k_values))
    sum_recalls = np.zeros(len(k_values))
    all_descriptors = []
    
    # Iterate through batches in the dataloader
    for batch_idx, batch in enumerate(dataloader):
        images, _ = batch
    
        # Calculate descriptors using the model
        descriptors = model(images.to(device)).cpu().detach().numpy().astype(np.float32)

        # Add descriptors to the list of concatenated descriptors
        all_descriptors.append(descriptors)  
        concatenated_descriptors = np.concatenate(all_descriptors, axis=0) 
        
    # Separate descriptors into database and query descriptors
    database_descriptors = concatenated_descriptors[: dataset.database_num ]
    queries_descriptors = concatenated_descriptors[dataset.database_num :]
    
    # Calculate recall using the recall function
    recalls = recall(dataset, database_descriptors, queries_descriptors, k_values, print_predictions, number_of_images_per_epoch)
    
    # Print recall values for specified k values
    # print(f'R@{k_values[0]}: {recalls[0]:.6f} ; R@{k_values[1]}: {recalls[1]:.6f} ; R@{k_values[2]}: {recalls[2]:.6f};')
    
    return recalls