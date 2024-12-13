import os

class TagManager:
    def __init__(self):
        self.image_tags = {}  # {image_path: tags_string}

    def load_tags(self, image_path):
        """Load tags from associated text file"""
        tag_path = os.path.splitext(image_path)[0] + '.txt'
        if os.path.exists(tag_path):
            try:
                with open(tag_path, 'r', encoding='utf-8') as f:
                    tags = f.read().strip()
                    self.image_tags[image_path] = tags
            except Exception as e:
                print(f"Error loading tags for {image_path}: {e}")
                self.image_tags[image_path] = ""
        else:
            self.image_tags[image_path] = ""

    def get_tags(self, image_path):
        """Get tags for an image"""
        return self.image_tags.get(image_path, "")

    def has_tags(self, image_path, search_tags):
        """Check if image has any of the specified tags"""
        image_tags_str = self.get_tags(image_path)
        if not image_tags_str:
            return False
            
        image_tags = {tag.strip().lower() for tag in image_tags_str.split(',') if tag.strip()}
        search_tags = {tag.strip().lower() for tag in search_tags if tag.strip()}
        
        # Return True if there's any intersection between the tag sets
        return bool(image_tags & search_tags)

    def clear(self):
        """Clear all loaded tags"""
        self.image_tags.clear()