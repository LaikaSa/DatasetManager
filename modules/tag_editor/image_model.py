from dataclasses import dataclass
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from collections import Counter
import os

@dataclass
class ImageData:
    path: str
    tags: set
    thumbnail: QPixmap = None

class ImageModel:
    def __init__(self):
        self.images = []
        self.thumbnail_size = 150

    def load_directory(self, directory):
        self.images.clear()
        
        for file in os.listdir(directory):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(directory, file)
                tag_path = os.path.splitext(image_path)[0] + '.txt'
                
                # Load tags
                try:
                    with open(tag_path, 'r', encoding='utf-8') as f:
                        tags = {tag.strip().lower() 
                               for tag in f.read().split(',') 
                               if tag.strip()}
                except:
                    tags = set()

                # Create thumbnail
                thumbnail = self.create_thumbnail(image_path)
                if thumbnail:
                    self.images.append(ImageData(image_path, tags, thumbnail))

    def create_thumbnail(self, image_path):
        image = QImage(image_path)
        if image.isNull():
            return None
            
        scaled = image.scaled(
            self.thumbnail_size, self.thumbnail_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        return QPixmap.fromImage(scaled)

    def get_tag_frequencies(self):
        counter = Counter()
        for image in self.images:
            counter.update(image.tags)
        return counter

    def filter_images(self, tags=None, combine_logic="AND", filter_logic="POSITIVE"):
        if not tags:
            return self.images

        # First apply AND/OR logic
        if combine_logic == "AND":
            matching_images = [
                img for img in self.images
                if tags.issubset(img.tags)
            ]
        else:  # OR logic
            matching_images = [
                img for img in self.images
                if bool(tags & img.tags)
            ]

        # Then apply POSITIVE/NEGATIVE logic
        if filter_logic == "POSITIVE":
            return matching_images
        else:  # NEGATIVE logic
            return [img for img in self.images if img not in matching_images]

    def clear(self):
        self.images.clear()