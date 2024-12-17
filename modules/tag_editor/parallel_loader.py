from multiprocessing import Pool, cpu_count
from pathlib import Path
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import os
import numpy as np
from PIL import Image
import time  # Add this import

def process_single_image(args):
    """Process a single image and its tags (runs in worker process)"""
    image_path, thumbnail_size = args
    try:
        # Use PIL instead of QImage for parallel processing
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate new size maintaining aspect ratio
            ratio = min(thumbnail_size / img.width, thumbnail_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            
            # Resize image
            thumbnail = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array for transfer
            img_array = np.array(thumbnail)

        # Read tags
        tag_path = str(Path(image_path).with_suffix('.txt'))
        tags = set()
        if os.path.exists(tag_path):
            with open(tag_path, 'r', encoding='utf-8') as f:
                tags = {tag.strip().lower() for tag in f.read().split(',') if tag.strip()}

        return {
            'path': image_path,
            'array': img_array,
            'tags': tags
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

class ParallelLoader:
    def __init__(self, thumbnail_size=150):
        self.thumbnail_size = thumbnail_size
        self.pool = None

    def start_pool(self):
        if self.pool is None:
            self.pool = Pool(processes=cpu_count())

    def stop_pool(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def load_images(self, directory):
        """Load images and tags in parallel"""
        try:
            print("\nStarting parallel loading process...")
            self.start_pool()
            
            # Get all image files
            file_scan_start = time.time()
            valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
            image_paths = [
                str(p) for p in Path(directory).glob('*.*')
                if p.suffix.lower() in valid_extensions
            ]
            file_scan_end = time.time()
            print(f"File scanning time: {file_scan_end - file_scan_start:.2f} seconds")
            print(f"Found {len(image_paths)} images")
            
            # Prepare arguments
            args = [(path, self.thumbnail_size) for path in image_paths]
            
            # Process images in parallel
            parallel_start = time.time()
            results = self.pool.map(process_single_image, args)
            parallel_end = time.time()
            print(f"Parallel processing time: {parallel_end - parallel_start:.2f} seconds")
            
            # Convert results to QPixmap in main thread
            conversion_start = time.time()
            processed_images = []
            successful = 0
            failed = 0
            
            for result in results:
                if result is None:
                    failed += 1
                    continue
                    
                try:
                    # Convert numpy array to QImage
                    height, width, channel = result['array'].shape
                    bytes_per_line = 3 * width
                    
                    q_img = QImage(
                        result['array'].data,
                        width,
                        height,
                        bytes_per_line,
                        QImage.Format_RGB888
                    )
                    
                    # Convert to QPixmap
                    thumbnail = QPixmap.fromImage(q_img)
                    
                    processed_images.append({
                        'path': result['path'],
                        'thumbnail': thumbnail,
                        'tags': result['tags']
                    })
                    successful += 1
                except Exception as e:
                    print(f"Error converting image {result['path']}: {e}")
                    failed += 1
            
            conversion_end = time.time()
            print(f"\nConversion Summary:")
            print(f"Successful conversions: {successful}")
            print(f"Failed conversions: {failed}")
            print(f"Conversion time: {conversion_end - conversion_start:.2f} seconds")
            
            return processed_images
                
        except Exception as e:
            print(f"Error in parallel loading: {e}")
            return []
        finally:
            self.stop_pool()