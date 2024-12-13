from enum import Enum

class TagFilter:
    class Logic(Enum):
        NONE = 0
        AND = 1
        OR = 2

    class Mode(Enum):
        NONE = 0
        INCLUSIVE = 1
        EXCLUSIVE = 2

    def __init__(self, tags=None, logic=Logic.OR, mode=Mode.INCLUSIVE):
        self.tags = set(tags) if tags else set()
        self.logic = logic
        self.mode = mode

    def matches(self, image_tags):
        """Check if image tags match the filter criteria"""
        if not self.tags or self.logic == TagFilter.Logic.NONE or self.mode == TagFilter.Mode.NONE:
            return True

        image_tagset = set(tag.strip().lower() for tag in image_tags.split(',') if tag.strip())
        filter_tagset = set(tag.strip().lower() for tag in self.tags)

        if self.logic == TagFilter.Logic.AND:
            if self.mode == TagFilter.Mode.INCLUSIVE:
                return image_tagset.issuperset(filter_tagset)
            else:  # EXCLUSIVE
                return not image_tagset.issuperset(filter_tagset)
        else:  # OR
            if self.mode == TagFilter.Mode.INCLUSIVE:
                return bool(image_tagset & filter_tagset)
            else:  # EXCLUSIVE
                return not bool(image_tagset & filter_tagset)