import torch
import numpy as np
from PIL import Image
import cv2

class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.preprocess_image(image)
        except Exception as e:
            logger.error(f"Could not load image path: {img_path}, error: {e}")
            return None

        return (image, img_path)

    def preprocess_image(self, image):
        image = np.array(image)
        image = image[:, :, ::-1]  # RGB->BGR

        # pad to square
        size = max(image.shape[0:2])
        pad_x = size - image.shape[1]
        pad_y = size - image.shape[0]
        pad_l = pad_x // 2
        pad_t = pad_y // 2
        image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), 
                      mode="constant", constant_values=255)

        interp = cv2.INTER_AREA if size > 448 else cv2.INTER_LANCZOS4
        image = cv2.resize(image, (448, 448), interpolation=interp)

        image = image.astype(np.float32)
        return image
    
def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch