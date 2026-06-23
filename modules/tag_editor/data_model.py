from dataclasses import dataclass
from pathlib import Path
from typing import Set, Dict, List
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from collections import Counter
from .parallel_loader import ParallelLoader
import time

@dataclass
class ImageData:
    path: str
    tags: list
    thumbnail: QPixmap = None
    modified: bool = False

class DataModel:
    def __init__(self):
        self.images: Dict[str, ImageData] = {}
        self.tag_frequencies = Counter()
        self.modified_files: Set[str] = set()
        self.parallel_loader = ParallelLoader()

    def filter_images(self, tags: Set[str], combine_logic: str = "AND", 
                     filter_logic: str = "POSITIVE") -> List[str]:
        """Filter images based on tags and logic"""
        print(f"Filtering with {len(tags)} tags using {combine_logic} logic and {filter_logic} mode")
        print(f"Tags to filter: {tags}")
        
        if not tags:
            return list(self.images.values())

        matching_images = []
        for image_data in self.images.values():
            matches = False
            if combine_logic == "AND":
                matches = tags.issubset(image_data.tags)
            else:  # OR
                matches = bool(tags & set(image_data.tags))

            if filter_logic == "POSITIVE":
                if matches:
                    matching_images.append(image_data)
            else:  # NEGATIVE
                if not matches:
                    matching_images.append(image_data)

        print(f"Found {len(matching_images)} matching images with {filter_logic} logic")
        return matching_images

    def load_directory(self, directory: str, use_parallel: bool = False) -> None:
        print(f"\nStarting directory load: {directory}")
        print(f"Using parallel loading: {use_parallel}")
        
        start_time = time.time()
        self.clear()

        if use_parallel:
            # Use parallel loading
            parallel_start = time.time()
            results = self.parallel_loader.load_images(directory)
            parallel_end = time.time()
            print(f"Parallel processing time: {parallel_end - parallel_start:.2f} seconds")

            # Process results
            process_start = time.time()
            for result in results:
                image_path = result['path']
                self.images[image_path] = ImageData(
                    image_path,
                    result['tags'],
                    result['thumbnail']
                )
                self.tag_frequencies.update(result['tags'])
            process_end = time.time()
            print(f"Results processing time: {process_end - process_start:.2f} seconds")

        else:
            # Sequential loading
            files = list(Path(directory).glob("*.*"))
            total_files = len([f for f in files 
                             if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}])
            print(f"Found {total_files} image files")

            processed = 0
            sequential_start = time.time()
            
            for file_path in files:
                if file_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}:
                    image_path = str(file_path)
                    tag_path = file_path.with_suffix('.txt')
                    
                    # Load tags (preserve order, drop duplicates)
                    tags = []
                    if tag_path.exists():
                        try:
                            with open(tag_path, 'r', encoding='utf-8') as f:
                                seen = set()
                                for tag in f.read().split(','):
                                    tag = tag.strip().lower()
                                    if tag and tag not in seen:
                                        seen.add(tag)
                                        tags.append(tag)
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
                        
                        # Store image data
                        self.images[image_path] = ImageData(
                            image_path,
                            tags,
                            thumbnail
                        )
                        self.tag_frequencies.update(tags)
                        
                        processed += 1
                        if processed % 100 == 0:  # Progress update every 100 images
                            print(f"Processed {processed}/{total_files} images...")
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")

            sequential_end = time.time()
            print(f"Sequential processing time: {sequential_end - sequential_start:.2f} seconds")

        end_time = time.time()
        total_time = end_time - start_time
        
        print("\nLoading Summary:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Images loaded: {len(self.images)}")
        print(f"Unique tags: {len(self.tag_frequencies)}")
        print(f"Average time per image: {(total_time / len(self.images) if self.images else 0):.3f} seconds")

    def remove_tags(self, tags_to_remove: Set[str]) -> None:
        print(f"Removing tags: {tags_to_remove}")
        for image_data in self.images.values():
            if set(image_data.tags) & tags_to_remove:  # If there are tags to remove
                image_data.tags = [t for t in image_data.tags if t not in tags_to_remove]
                image_data.modified = True
                self.modified_files.add(image_data.path)
        
        # Update tag frequencies
        self.update_tag_frequencies()

    def update_tag_frequencies(self) -> None:
        self.tag_frequencies.clear()
        for image_data in self.images.values():
            self.tag_frequencies.update(image_data.tags)
        print(f"Updated tag frequencies: {len(self.tag_frequencies)} unique tags")

    def update_image_tags(self, image_path: str, new_tags: set):
        """Update tags for an image"""
        print(f"Updating tags for {image_path}")
        print(f"New tags: {new_tags}")
        if image_path in self.images:
            image_data = self.images[image_path]
            image_data.tags = new_tags
            image_data.modified = True
            self.modified_files.add(image_path)
            self.update_tag_frequencies()

    def save_changes(self, create_backup: bool = False) -> tuple[int, int]:
        saved_count = 0
        
        for image_path in self.modified_files:
            image_data = self.images[image_path]
            txt_path = Path(image_path).with_suffix('.txt')

            # Create backup if requested
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

            # Write new tags to file
            try:
                # Sort tags and join with commas
                tag_text = ', '.join(image_data.tags)
                print(f"Writing tags to {txt_path}: {tag_text}")  # Debug print
                
                # Write to file
                txt_path.write_text(tag_text, encoding='utf-8')
                image_data.modified = False
                saved_count += 1
                print(f"Successfully saved {txt_path}")  # Debug print
            except Exception as e:
                print(f"Failed to save {txt_path}: {e}")

        total_modified = len(self.modified_files)
        self.modified_files.clear()
        print(f"Saved {saved_count}/{total_modified} files")  # Debug print
        return saved_count, total_modified

    def clear(self) -> None:
        self.images.clear()
        self.tag_frequencies.clear()
        self.modified_files.clear()

    def add_image(self, item):
        """Add a processed image to the model"""
        self.images[item['path']] = ImageData(
            item['path'],
            item['tags'],
            item['thumbnail']
        )
        self.tag_frequencies.update(item['tags'])