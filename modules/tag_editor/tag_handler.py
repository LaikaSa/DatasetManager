from collections import Counter

def load_tags(tag_path):
    try:
        with open(tag_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except:
        return ''

def count_tags(image_tags_list):
    counter = Counter()
    for tags in image_tags_list:
        if tags:
            for tag in tags.lower().split(','):
                tag = tag.strip()
                if tag:
                    counter[tag] += 1
    return counter