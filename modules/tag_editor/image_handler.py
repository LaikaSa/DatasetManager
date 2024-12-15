import os
from .image_viewer import ThumbnailLabel
from . import tag_handler

def load_from_directory(directory):
    images = []
    tags_list = []
    
    for file in os.listdir(directory):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(directory, file)
            tag_path = os.path.splitext(image_path)[0] + '.txt'
            
            # Load tags
            tags = tag_handler.load_tags(tag_path)
            tags_list.append(tags)
            
            # Create thumbnail
            thumbnail = ThumbnailLabel(image_path, tags)
            images.append(thumbnail)
            
    return images, tags_list