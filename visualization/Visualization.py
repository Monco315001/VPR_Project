# Import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw, ImageFont
from skimage.transform import rescale


# Function to denormalize image for visualization
def denormalize(image):
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image


def print_sample_dataset(dataset_type, num_batch, num_images_per_batch, dataloader_train, dataloader_test):
    """
    Prints a sample of images from the dataset.

    Args:
        dataset_type (str): The type of dataset ('train', 'val', 'test').
        num_batch (int): The number of batches to display.
        num_images_per_batch (int): The number of images to display per batch.
    """
    # Train images
    if dataset_type == 'train':
        batch_count = 0
        for images, labels, place_id in dataloader_train:
            print(f"Batch:{batch_count}")
            # Make sure not to exceed the number of images in the batch
            num_images = min(num_images_per_batch, len(images))
            # Denormalize and display images in the batch
            for i in range(num_images):
                label = place_id[i]
                for j in range(4):
                    img = denormalize(images[i][j])  # Denormalize the i-th images
                    plt.figure()
                    plt.imshow(img)
                    plt.title(f'ID: {label}')
                    plt.show()

            batch_count += 1

            # To limit the number of displayed batches
            if batch_count >= num_batch:  # Show only the first 10 batches
                break

    # Val and test images
    elif dataset_type == 'test' or dataset_type == 'val':
        batch_count = 0
        for images, labels in dataloader_test: 
            print(f"Batch:{batch_count}")
            # Make sure not to exceed the number of images in the batch
            num_images = min(num_images_per_batch, len(images))
            # Denormalize and display images in the batch
            for i in range(num_images):
                img = denormalize(images[i])  # Denormalize the i-th image
                plt.figure()
                plt.imshow(img)
                plt.show()

            batch_count += 1

            # To limit the number of displayed batches
            if batch_count >= num_batch:  # Show only the first 10 batches
                break

    else:
        raise ValueError("Error: possible dataset types are 'train', 'val', and 'test'")
    

def write_labels_to_image(labels=["text1", "text2"], space=150, fontsize=100, Width=1024, Total_height=350):
    """
    Create an image with labels centered horizontally.

    Args:
        labels (list): List of strings to be used as labels. Default is ["text1", "text2"].

    Returns:
        np.array: Image array with labels.
    """
    # Load the font
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)
    
    # Calculate the width of the resulting image
    img_width = Width * len(labels) + space * (len(labels) - 1)
    img_height = Total_height
    
    # Create a new image with white background
    background_color = (1, 1, 1)  # White color in RGB
    img = Image.new('RGB', (img_width, img_height), background_color)
    draw = ImageDraw.Draw(img)
    
    # Draw each label centered horizontally
    for i, text in enumerate(labels):
        _, _,text_width, text_height = draw.textbbox((0,0), text, font=font)
        x = (Width + space) * i + Width //2 - text_width // 2
        y = (img_height - text_height) // 2
        draw.text((x, y), text, fill=(0, 0, 0), font=font)  # Black text color
    
    return np.array(img)


def draw_box(img, c): 
    """
    Draw a colored box around the image.

    Args:
        img (numpy.ndarray): Input image array.
        color (tuple): RGB tuple for the color of the box.

    Returns:
        numpy.ndarray: Image with a colored box.
    """
    thickness=5
    height, width = img.shape[:2]
    
    # Draw lines to create a rectangle
    cv2.line(img, (0, 0), (0, height), c, thickness)  # left vertical line
    cv2.line(img, (0, height), (width, height), c, thickness)  # upper horizontal line
    cv2.line(img, (width, height), (width, 0), c, thickness)  # right vertical line
    cv2.line(img, (width, 0), (0, 0), c, thickness)  # lower horizontal line
    
    return img


def print_preds(predictions, test_dataset,number_of_images_per_epoch, space=150, Width=1024, Height=1024):
    """
    Print predictions with images and labels in a collage format.

    Args:
        predictions (list): List of predictions for each query.
        test_dataset (object): Dataset object containing queries and database paths.
        number_of_images_per_epoch (int): Maximum number of images to display per epoch.
    """
    x = 0 
    positives_per_query = test_dataset.get_positives()  # Retrieve true positives
    
    # Iterate through each query and its predictions
    for q_idx, preds in enumerate(predictions):
        if x >= number_of_images_per_epoch:
            break
            
        query_path = test_dataset.queries_paths[q_idx]
        list_of_images_paths = [query_path]  # start with the query path
        
        # List of prediction types (None for query, True for correct, False for wrong)
        preds_types = [None]
        
        # Iterate through predictions and determine correctness
        for _ , pred in enumerate(preds):
            pred_path = test_dataset.database_paths[pred]
            list_of_images_paths.append(pred_path)  # list of query path + paths of all its predictions
            
            # Check if the prediction is correct, comparing to true positives
            if pred in positives_per_query[q_idx]: 
                type_of_pred = True
            else:
                type_of_pred = False
                
            preds_types.append(type_of_pred)
        
        # Generate labels for the collage
        labels = ["Query"] + [f"Prediction{i} - {type_of_pred}" for i, type_of_pred in enumerate(preds_types[1:])]
        num_images = len(list_of_images_paths)
        color=[]
        
        # Load images and apply colored boxes (green for correct, red for wrong)
        images = [np.array(Image.open(path)) for path in list_of_images_paths]
        for img, correct in zip(images, preds_types):
            if correct is not None:
                if correct:
                    color = (0, 255, 0)  # Green for correct
                else:
                    color = (255, 0, 0)  # Red for wrong
            draw_box(img, color)
        
        # Concatenate images horizontally with padding and scaling
        concat_image = np.ones([Height, (num_images*Width)+((num_images-1)*space), 3])
        rescaleds = [rescale(i, [min(Height/i.shape[0], Width/i.shape[1]), min(Height/i.shape[0], Width/i.shape[1]), 1]) for i in images]
        
        # Zero padding needed to center the image 
        for i, image in enumerate(rescaleds):
            pad_width = (Width - image.shape[1] + 1) // 2
            pad_height = (Height - image.shape[0] + 1) // 2
            image = np.pad(image, [[pad_height, pad_height], [pad_width, pad_width], [0, 0]], constant_values=1)[:Height, :Width]
            concat_image[: , i*(Width+space) : i*(Width+space)+Width] = image
        
        # Create final collage image with labels
        labels_image = write_labels_to_image(labels)
        final_image = np.concatenate([labels_image, concat_image])
        final_image = Image.fromarray((final_image*255).astype(np.uint8))
        
        # Display the final image using matplotlib
        plt.figure()
        plt.imshow(final_image)
        plt.axis('off')
        plt.show()
        
        x += 1
