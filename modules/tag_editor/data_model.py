from dataclasses import dataclass
from pathlib import Path
from typing import Set, Dict, List, Tuple, Union
from PySide6.QtGui import QPixmap
from collections import Counter
from . import engine

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

    def filter_images(self, tags: Set[str], combine_logic: str = "AND",
                     filter_logic: str = "POSITIVE") -> List[ImageData]:
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

    def remove_tags(self, tags_to_remove: Set[str]) -> None:
        print(f"Removing tags: {tags_to_remove}")
        for image_data in self.images.values():
            new_tags = engine.apply_remove(image_data.tags, tags_to_remove)
            if new_tags is not None:  # If there are tags to remove
                image_data.tags = new_tags
                image_data.modified = True
                self.modified_files.add(image_data.path)
        
        # Update tag frequencies
        self.update_tag_frequencies()

    def replace_or_add_tags(self, tags_to_replace: Set[str], new_tags: List[str],
                                 positions: List[Union[str, Tuple[str, int]]]) -> None:
            """
            tags_to_replace: set of tags to remove first (empty set = pure add, no removal).
            new_tags: ordered list of tags to insert.
            positions: list made of 'top', 'middle', 'bottom', and/or ('custom', n).
                       If multiple are given, new_tags are cloned and inserted at each spot.
            """
            if not new_tags:
                return

            for image_data in self.images.values():
                tags, changed = engine.apply_replace_add(
                    image_data.tags, tags_to_replace, new_tags, positions)
                if changed:
                    image_data.tags = tags
                    image_data.modified = True
                    self.modified_files.add(image_data.path)

            self.update_tag_frequencies()

    def update_tag_frequencies(self) -> None:
        self.tag_frequencies.clear()
        for image_data in self.images.values():
            self.tag_frequencies.update(image_data.tags)
        print(f"Updated tag frequencies: {len(self.tag_frequencies)} unique tags")

    def update_image_tags(self, image_path: str, new_tags):
        """Update tags for an image.

        Frequencies are adjusted incrementally instead of recomputing the
        Counter over the whole dataset - this method fires on every caption
        edit keystroke, and a full recompute per keystroke is O(dataset)."""
        if image_path in self.images:
            image_data = self.images[image_path]
            old_tags = image_data.tags
            image_data.tags = list(new_tags)
            image_data.modified = True
            self.modified_files.add(image_path)

            for tag in old_tags:
                if self.tag_frequencies[tag] > 0:
                    self.tag_frequencies[tag] -= 1
            for tag in image_data.tags:
                self.tag_frequencies[tag] += 1
            # Drop zeroed entries so the filter list stays clean
            for tag in [t for t, c in self.tag_frequencies.items() if c <= 0]:
                del self.tag_frequencies[tag]

    def save_changes(self, create_backup: bool = False) -> tuple[int, int]:
        saved_count = 0
        
        for image_path in self.modified_files:
            image_data = self.images[image_path]
            txt_path = Path(image_path).with_suffix('.txt')

            # Create backup if requested
            if create_backup and txt_path.exists():
                try:
                    engine.backup_sidecar(txt_path)
                except Exception as e:
                    print(f"Failed to create backup for {txt_path}: {e}")
                    continue

            # Write new tags to file
            try:
                # Join tags with commas (helper shared with the CLI engine)
                tag_text = engine.serialize_tags(image_data.tags)
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
