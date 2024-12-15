from dataclasses import dataclass
from pathlib import Path
from typing import Set, Dict, List
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from collections import Counter

@dataclass
class ImageData:
    path: str
    tags: set
    thumbnail: QPixmap = None
    modified: bool = False

class DataModel:
    def __init__(self):
        self.images: Dict[str, ImageData] = {}
        self.tag_frequencies = Counter()
        self.modified_files: Set[str] = set()

    def load_directory(self, directory: str) -> None:
        print(f"Loading directory: {directory}")
        self.clear()
        
        for file_path in Path(directory).glob("*.*"):
            if file_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}:
                image_path = str(file_path)
                tag_path = file_path.with_suffix('.txt')
                
                # Load tags
                tags = set()
                if tag_path.exists():
                    try:
                        with open(tag_path, 'r', encoding='utf-8') as f:
                            tags = {tag.strip().lower() 
                                  for tag in f.read().split(',') 
                                  if tag.strip()}
                    except Exception as e:
                        print(f"Error loading tags for {image_path}: {e}")

                # Create thumbnail
                try:
                    thumbnail = QPixmap(image_path)
                    thumbnail = thumbnail.scaled(
                        150, 150,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                except Exception as e:
                    print(f"Error creating thumbnail for {image_path}: {e}")
                    continue

                # Store image data
                self.images[image_path] = ImageData(image_path, tags, thumbnail)
                self.tag_frequencies.update(tags)

        print(f"Loaded {len(self.images)} images with {len(self.tag_frequencies)} unique tags")

    def filter_images(self, tags: Set[str], combine_logic: str = "AND", 
                    filter_logic: str = "POSITIVE") -> List[str]:
        print(f"Filtering with {len(tags)} tags using {combine_logic} logic")
        print(f"Tags to filter: {tags}")
        
        if not tags:
            return list(self.images.keys())

        matching_images = []
        for path, image_data in self.images.items():
            matches = False
            if combine_logic == "AND":
                matches = tags.issubset(image_data.tags)
            else:  # OR
                matches = bool(tags & image_data.tags)

            if filter_logic == "POSITIVE":
                if matches:
                    matching_images.append(path)
            else:  # NEGATIVE
                if not matches:
                    matching_images.append(path)

        print(f"Found {len(matching_images)} matching images")
        return matching_images

    def remove_tags(self, tags_to_remove: Set[str]) -> None:
        print(f"Removing tags: {tags_to_remove}")
        for image_data in self.images.values():
            if image_data.tags & tags_to_remove:  # If there are tags to remove
                image_data.tags -= tags_to_remove
                image_data.modified = True
                self.modified_files.add(image_data.path)
        
        # Update tag frequencies
        self.update_tag_frequencies()

    def update_tag_frequencies(self) -> None:
        self.tag_frequencies.clear()
        for image_data in self.images.values():
            self.tag_frequencies.update(image_data.tags)
        print(f"Updated tag frequencies: {len(self.tag_frequencies)} unique tags")

    def save_changes(self, create_backup: bool = False) -> tuple[int, int]:
        saved_count = 0
        
        for image_path in self.modified_files:
            image_data = self.images[image_path]
            txt_path = Path(image_path).with_suffix('.txt')

            if create_backup and txt_path.exists():
                try:
                    backup_num = 0
                    while True:
                        backup_path = txt_path.with_suffix(f'.{backup_num:03d}')
                        if not backup_path.exists():
                            txt_path.rename(backup_path)
                            break
                        backup_num += 1
                except Exception as e:
                    print(f"Failed to create backup for {txt_path}: {e}")
                    continue

            try:
                txt_path.write_text(', '.join(sorted(image_data.tags)), encoding='utf-8')
                image_data.modified = False
                saved_count += 1
            except Exception as e:
                print(f"Failed to save {txt_path}: {e}")

        total_modified = len(self.modified_files)
        self.modified_files.clear()
        return saved_count, total_modified

    def clear(self) -> None:
        self.images.clear()
        self.tag_frequencies.clear()
        self.modified_files.clear()