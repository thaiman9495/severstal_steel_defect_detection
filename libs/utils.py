import numpy as np

def rle_to_mask(rle_mask: str, image_shape: tuple):
    """This function converts a string of run-lenth-encoding masks to an image mask

    Args:
        rle_mask (str): Run-length-encoding mask
        image_shape (tuple): (height, width)

    Returns:
        numpy.ndarray: 1 - mask, 0 - background
    """
    
    IMG_HEIGHT, IMG_WIDTH = image_shape[0], image_shape[1]
    
    # Break rle mask into two list of "start" and "length" values
    s = rle_mask.split()
    starts = np.asarray(s[0:][::2], dtype=int)  # List slicing technique
    lengths = np.asarray(s[1:][::2], dtype=int) # List slicing technique

    # Compute the index of the end pixel for each rle pair value
    starts -= 1
    ends = starts + lengths

    # Compute mask
    mask = np.zeros(IMG_HEIGHT * IMG_WIDTH, dtype=np.uint8)
    for start_id, end_id in zip(starts, ends):
        mask[start_id:end_id] = 1

    # Beacause the pixels are numbered from top to bottom, then left to right:
    # 1 is pixel (1,1), 2 is pixel (2,1), etc, we need to reshape the flatten 
    # mask to the one of shape (IMG_WIDTH, IMG_HEIGHT) and then tranpose it.
    mask = mask.reshape(IMG_WIDTH, IMG_HEIGHT).T

    return mask