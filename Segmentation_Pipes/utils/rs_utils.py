import torch
import numpy as np
import matplotlib.pyplot as plt

def torch_denormalize(img):
    
    img[0,:]  =  (img[0,:]* 0.229) + 0.485
    img[1,:]  =  (img[1,:]* 0.224) + 0.456
    img[2,:]  =  (img[2,:]* 0.225) + 0.406
    
    return img   




def torch_display(image,batch,denormal):
    
    '''
    input shape : batch or not  
    
    torch.Size([batch, channel, 256, 256])
    
    '''
    img = image.clone() 
    if batch:
        img = img[0,:]
    else:
        pass
    
    #----- denormalizing func
    if denormal:
        # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        img = torch_denormalize(img)
    #-----
             
    img = img.permute(1,2,0)
    img = np.asarray(img)   
    
    fig_size= (10,10)
    plt.figure(figsize=fig_size)
    plt.imshow(img)
    
    
#--
import logging
import os

# Function to configure the logger
def setup_logger(name, log_file, level=logging.INFO):
    """
    Set up a logger with the specified name, log file, and log level.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create file handler to save logs to a file
    file_handler = logging.FileHandler(log_file, mode='a')  # 'a' for append mode
    file_handler.setLevel(level)
    
    # Create console handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Define a log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger