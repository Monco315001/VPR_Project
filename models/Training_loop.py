# Import libraries
import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# KNN search for the pre-miner
def knn_search(proxies, proxy_labels):
    """
    Performs K-nearest neighbors (KNN) search on the given proxies to create informative batches.

    Args:
        proxies (np.ndarray): Array of proxy vectors.
        proxy_labels (np.ndarray): Array of labels corresponding to the proxy vectors.

    Returns:
        list: A list of informative batches, each containing the labels of the k-nearest neighbors.
    """
    informative_batches = []
    k = 60
    while len(proxies) >= k:
        # Create an index object with a flat L2 distance metric
        faiss_index = faiss.IndexFlatL2(proxies.shape[1])

        # Add the vectors to the index
        faiss_index.add(proxies)

        # Define a query vector (the first proxy vector in the list)
        query_vector = proxies[0]
        query_vector = np.reshape(query_vector, (1, -1))
        
        # Perform the KNN search to find the k-nearest neighbors
        distances, indices = faiss_index.search(query_vector, k)
        
        # Extract the indices of the k-nearest neighbors
        indices_list = [indices[0][:][i] for i in range(k)]
        
        # Extract the labels of the k-nearest neighbors
        informative_batches_labels = [proxy_labels[idx] for idx in indices_list]
        
        # Append the batch of labels to the informative batches list
        informative_batches.append(informative_batches_labels)
        
        # Remove the found proxies and their labels from the arrays
        proxies = np.delete(proxies, indices_list, axis=0)
        proxy_labels = np.delete(proxy_labels, indices_list, axis=0)
    
    return informative_batches


# Training Loop 
def training_loop(epoch, model, dataset, dataloader, criterion, optimizer, miner = None, pre_miner = None):
    """
    Training loop for the model.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): Model to be trained.
        dataloader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for training.
        miner (callable, optional): Online mining strategy. Default is None.
        pre_miner (callable, optional): Pre-mining strategy. Default is None.
    """
    
    model.train()
    inf_batch_count = 0  # Counter for informative batches
    train_loss = 0
    
    # In the first epoch or if no pre-miner is provided, don't use the proxy to improve efficiency
    if epoch == 1 or pre_miner is None:  
        
        if pre_miner is not None:  # Initialize variables for creating proxy vectors
            global informative_batches
            informative_batches = []
            proxy_labels = []
            proxies = []
            
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            images, labels, _ = batch
            
            # Reshape images and labels
            num_places, num_images_per_place, C, H, W = images.shape
            images = images.view(num_places * num_images_per_place, C, H, W)
            labels = labels.view(num_places * num_images_per_place)
            
            # Get model descriptors for the images
            descriptors = model(images.to(device)).cpu()

            if pre_miner is not None:  # Create proxy vectors
                with torch.nograd():   # Exclude operations from the backpropagation
                    num_tensori, *_ = descriptors.shape
                    for i in range(0, num_tensori - 4 + 1, 4):
                        place_images = descriptors[i:i+4]
                        proxy = place_images.mean(dim=0).tolist()
                        proxies.append(proxy)
                        proxy_labels.append(int(labels[i]))

            # MINING: Mine the pairs/triplets if there is an online mining strategy
            if miner is not None:
                miner_outputs = miner(descriptors, labels.to(device))
                loss = criterion(descriptors, labels.to(device), miner_outputs)

                # Calculate the % of trivial pairs/triplets which do not contribute to the loss value
                nb_samples = descriptors.shape[0]
                nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
                batch_acc = 1.0 - (nb_mined / nb_samples)

            else:  # No online mining
                loss = criterion(descriptors, labels.to(device))
                batch_acc = 0.0
            
            # Backpropagation and optimization
            loss.backward() 
            optimizer.step()
            train_loss += loss.item()

        if pre_miner is not None:  # Create informative batches
            proxies = np.asarray(proxies, dtype = np.float32)
            proxy_labels = np.asarray(proxy_labels, dtype = np.int32)
            informative_batches = knn_search(proxies, proxy_labels)
    
    # Use informative batches for training
    else:        
        # Initialize variables for creating proxy vectors
        proxy_labels = []
        proxies = []
        
        for batch in informative_batches:       
            optimizer.zero_grad()
            
            # Get images for the current batch
            images = [dataset.__getitem__(label)[0] for label in batch]
            images = torch.stack(images)
            dimensions = images.shape
            labels = [torch.tensor(label).repeat(4) for label in batch]
            labels = torch.stack(labels)
            
            # Reshape images and labels
            num_places, num_images_per_place, C, H, W = images.shape
            images = images.view(num_places * num_images_per_place, C, H, W)
            labels = labels.view(num_places * num_images_per_place)
            
            # Get model descriptors for the images
            descriptors = model(images.to(device)).cpu()
            
            # Create proxy vectors
            with torch.nograd():  # Exclude operations from the backpropagation
                num_tensori, *_ = descriptors.shape
                for i in range(0, num_tensori - 4 + 1, 4):
                    place_images = descriptors[i:i+4]
                    proxy = place_images.mean(dim=0).tolist()
                    proxies.append(proxy)
                    proxy_labels.append(int(labels[i]))

            # MINING: Mine the pairs/triplets if there is an online mining strategy
            if miner is not None:
                miner_outputs = miner(descriptors, labels.to(device))
                loss = criterion(descriptors, labels.to(device), miner_outputs)

                # Calculate the % of trivial pairs/triplets which do not contribute in the loss value
                nb_samples = descriptors.shape[0]
                nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
                batch_acc = 1.0 - (nb_mined / nb_samples)

            else:  # No online mining
                loss = criterion(descriptors, labels.to(device))
                batch_acc = 0.0
            
            # Backpropagation and optimization
            loss.backward() 
            optimizer.step()
            train_loss += loss.item()
            inf_batch_count += 1

        # Create informative batches
        proxies = np.asarray(proxies, dtype = np.float32)
        proxy_labels = np.asarray(proxy_labels, dtype = np.int32)
        informative_batches = knn_search(proxies, proxy_labels)
    
    # Calculate the average training loss
    train_loss = train_loss / len(dataloader)
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f}')
    