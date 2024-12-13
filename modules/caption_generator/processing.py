from PySide6.QtCore import QThread, Signal
import numpy as np
import os
import torch
from PIL import Image
from .utils import ProgressBar
from .data_loader import ImageLoadingPrepDataset, collate_fn_remove_corrupted
from modules.logger import setup_logger
logger = setup_logger()

class CaptionGeneratorThread(QThread):
    caption_generated = Signal(str, str)
    process_completed = Signal()
    error_occurred = Signal(str)

    def __init__(self, captioner, folder_path, include_rating=False, 
                 remove_underscore=True, recursive=False, 
                 undesired_tags=None, prefix_tags=None, append_tags=False,
                 thresh=0.35, general_threshold=0.35, character_threshold=0.35,
                 batch_size=1, worker_count=2):
        super().__init__()
        self.captioner = captioner
        self.folder_path = folder_path
        self.include_rating = include_rating
        self.remove_underscore = remove_underscore
        self.recursive = recursive
        self.undesired_tags = undesired_tags or set()
        self.prefix_tags = prefix_tags or []
        self.append_tags = append_tags
        self.thresh = thresh
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.batch_size = batch_size
        self.worker_count = worker_count

    def run(self):
        try:
            image_files = self.get_image_files(self.folder_path)
            total_files = len(image_files)
            
            # Basic info always shown
            logger.info(f"Starting caption generation for {len(image_files)} images")
            progress = ProgressBar(total_files, prefix='Processing: ')

            for i, image_path in enumerate(image_files, 1):
                try:
                    # Generate new caption
                    caption = self.captioner.generate_caption(
                        image_path,
                        include_rating=self.include_rating,
                        remove_underscore=self.remove_underscore,
                        undesired_tags=self.undesired_tags,
                        always_first_tags=self.prefix_tags,
                        thresh=self.thresh,
                        general_threshold=self.general_threshold,
                        character_threshold=self.character_threshold
                    )
                    
                    txt_path = os.path.splitext(image_path)[0] + '.txt'
                    
                    # Handle append mode
                    if self.append_tags and os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                existing_content = f.read().strip()
                            
                            existing_tags = [tag.strip() for tag in existing_content.split(',') if tag.strip()]
                            new_tags = [tag.strip() for tag in caption.split(',') if tag.strip()]
                            
                            # Only log detailed tag info in debug mode
                            if self.captioner.debug_mode:
                                logger.debug(f"\nAppending tags for {os.path.basename(image_path)}:")
                                logger.debug(f"  Existing tags: {existing_tags}")
                                logger.debug(f"  New tags: {new_tags}")
                            
                            combined_tags = []
                            seen = set()
                            
                            for tag in existing_tags:
                                if tag not in seen:
                                    combined_tags.append(tag)
                                    seen.add(tag)
                            
                            for tag in new_tags:
                                if tag not in seen:
                                    combined_tags.append(tag)
                                    seen.add(tag)
                            
                            caption = ', '.join(combined_tags)
                            
                            # Log final combined tags only in debug mode
                            if self.captioner.debug_mode:
                                logger.debug(f"  Final combined tags: {combined_tags}")
                            
                        except Exception as e:
                            logger.error(f"Error reading existing caption for {image_path}: {e}")
                    else:
                        # Log generated tags only in debug mode
                        if self.captioner.debug_mode:
                            logger.debug(f"\nGenerated caption for {os.path.basename(image_path)}:")
                            logger.debug(f"  Tags: {caption.split(', ')}")

                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(caption + '\n')
                    
                    # Basic progress info always shown
                    self.caption_generated.emit(image_path, caption)
                    progress.update(i)
                
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {str(e)}")
                    self.error_occurred.emit(f"Error processing {image_path}: {str(e)}")
                    continue

            # Basic completion info always shown
            logger.info("Caption generation completed")
            self.process_completed.emit()

        except Exception as e:
            error_msg = f"Process error: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def process_batch(self, batch_images):
        try:
            # Prepare batch data
            images = np.array([img for img, _ in batch_images])
            
            # Run inference on batch
            captions = []
            for image in images:
                caption = self.captioner.generate_caption(
                    image,
                    include_rating=self.include_rating,
                    remove_underscore=self.remove_underscore,
                    undesired_tags=self.undesired_tags,
                    always_first_tags=self.prefix_tags,
                    general_threshold=self.general_threshold,
                    character_threshold=self.character_threshold,
                    thresh=self.thresh
                )
                captions.append(caption)

            # Save captions
            for (_, image_path), caption in zip(batch_images, captions):
                txt_path = os.path.splitext(image_path)[0] + '.txt'
                
                if self.append_tags and os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            existing_content = f.read().strip()
                        
                        existing_tags = [tag.strip() for tag in existing_content.split(',') if tag.strip()]
                        new_tags = [tag.strip() for tag in caption.split(',') if tag.strip()]
                        
                        combined_tags = []
                        seen = set()
                        
                        for tag in existing_tags:
                            if tag not in seen:
                                combined_tags.append(tag)
                                seen.add(tag)
                        
                        for tag in new_tags:
                            if tag not in seen:
                                combined_tags.append(tag)
                                seen.add(tag)
                        
                        caption = ', '.join(combined_tags)
                        
                    except Exception as e:
                        self.error_occurred.emit(f"Error reading existing caption for {image_path}: {str(e)}")

                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(caption + '\n')
                
                self.caption_generated.emit(image_path, caption)

        except Exception as e:
            self.error_occurred.emit(f"Error processing batch: {str(e)}")

    def get_image_files(self, folder_path):
        image_files = []
        if self.recursive:
            # Walk through directory and subdirectories
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_files.append(os.path.join(root, file))
        else:
            # Only get files from the main directory
            image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                         if os.path.isfile(os.path.join(folder_path, f)) and
                         f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        return image_files

