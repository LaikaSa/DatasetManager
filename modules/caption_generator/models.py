import onnxruntime
import onnx
import numpy as np
import os
import multiprocessing
import pandas as pd
from PIL import Image
from modules.logger import setup_logger

logger = setup_logger()

class ImageCaptioner:
    MODELS = {
        'wd-eva02-large-tagger-v3': {
            'path': os.path.join('models', 'wd-eva02-large-tagger-v3'),  # Use os.path.join
            'type': 'eva02-v3',
            'repo_id': 'SmilingWolf/wd-eva02-large-tagger-v3'
        },
        'wd-swinv2-tagger-v3': {
            'path': os.path.join('models', 'wd-swinv2-tagger-v3'),
            'type': 'swinv2-v3',
            'repo_id': 'SmilingWolf/wd-swinv2-tagger-v3'
        },
        'wd-convnext-tagger-v3': {
            'path': os.path.join('models', 'wd-convnext-tagger-v3'),
            'type': 'convnext-v3',
            'repo_id': 'SmilingWolf/wd-convnext-tagger-v3'
        }
    }

    def __init__(self, model_name='wd-eva02-large-tagger-v3', debug_mode=False):
        self.debug_mode = debug_mode  # Store debug mode as instance variable
        print(f"Initializing ImageCaptioner with model: {model_name}")
        
        # Get root directory (two levels up from models.py)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.MODELS.keys())}")

        # Get model info
        model_info = self.MODELS[model_name]
        self.model_type = model_info['type']
        
        # Construct full paths using root directory
        model_dir = os.path.join(root_dir, model_info['path'])
        model_path = os.path.join(model_dir, "model.onnx")
        tags_path = os.path.join(model_dir, "selected_tags.csv")
        
        print(f"Looking for model at: {model_path}")
        print(f"Looking for tags at: {tags_path}")

        # Verify paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(tags_path):
            raise FileNotFoundError(f"Tags file not found at: {tags_path}")

        # Load model
        self.session = self.create_session(model_path)
        if self.session is None:
            raise Exception("Failed to create ONNX session")

        # Load tags
        tags_df = pd.read_csv(tags_path)
        self.load_tags(tags_df)
        
        # Get model input shape
        _, height, width, _ = self.session.get_inputs()[0].shape
        self.target_size = height
        print(f"Model input shape: {height}x{width}")

    def create_session(self, model_path):
        try:
            import onnx
            import onnxruntime as ort
            
            print(f"Loading ONNX model: {model_path}")
            model = onnx.load(model_path)
            input_name = model.graph.input[0].name
            self.input_name = input_name

            print(f"Available providers: {ort.get_available_providers()}")
            
            # Configure session options
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Set up providers
            providers = []
            provider_options = []
            
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
                provider_options.append({
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                })
            
            # Always add CPU provider
            providers.append("CPUExecutionProvider")
            provider_options.append({})
            
            print(f"Using providers: {providers}")
            print(f"With options: {provider_options}")
            
            # Create session
            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers,
                provider_options=provider_options
            )
            
            print(f"Session created successfully with active providers: {session.get_providers()}")
            return session

        except Exception as e:
            print(f"Error creating ONNX session: {e}")
            print(f"Exception type: {type(e)}")
            print(f"Exception args: {e.args}")
            
            try:
                print("Attempting fallback with minimal settings...")
                session = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider'],
                    provider_options=[{}]
                )
                print("Successfully created basic session")
                return session
            except Exception as fallback_e:
                print(f"Fallback also failed: {fallback_e}")
                return None
            
    def generate_caption(self, image_path, 
                        general_threshold=0.35, 
                        character_threshold=0.85,
                        thresh=0.35,
                        remove_underscore=True,
                        undesired_tags=None,
                        always_first_tags=None,
                        caption_separator=", ",
                        include_rating=False):
        try:
            # 1. Initial Setup and Logging
            if self.debug_mode:  # Use debug_mode instead of debug_checkbox
                logger.debug(f"Generating caption for {os.path.basename(image_path)}")
                logger.debug("Parameters:")
                logger.debug(f"  General threshold: {general_threshold}")
                logger.debug(f"  Character threshold: {character_threshold}")
                logger.debug(f"  Overall threshold: {thresh}")
                logger.debug(f"  Include rating: {include_rating}")
                logger.debug(f"  Remove underscore: {remove_underscore}")
            
            # Set thresholds
            general_threshold = general_threshold if general_threshold != thresh else thresh
            character_threshold = character_threshold if character_threshold != thresh else thresh
            
            # 2. Image Preparation
            if isinstance(image_path, str):
                image = self.prepare_image(image_path)
            else:
                image = image_path
                
            # 3. Model Inference
            label_name = self.session.get_outputs()[0].name
            preds = self.session.run([label_name], {self.input_name: image})[0]
            labels = list(zip(self.tag_names, preds[0].astype(float)))
            
            # 4. Tag Processing
            combined_tags = []
            
            # 4.1 Add Prefix Tags
            if always_first_tags:
                combined_tags.extend(always_first_tags)
                if self.debug_mode:
                    logger.debug(f"Added prefix tags: {always_first_tags}")
            
            # 4.2 Process Rating Tags
            if include_rating:
                ratings_names = [labels[i] for i in self.rating_indexes]
                rating = dict(ratings_names)
                rating_tag = max(rating.items(), key=lambda x: x[1])[0]
                if rating_tag not in combined_tags:
                    combined_tags.append(rating_tag)
                    if self.debug_mode:
                        logger.debug(f"Added rating tag: {rating_tag}")
            
            # 4.3 Process Character Tags
            character_names = [labels[i] for i in self.character_indexes]
            character_res = [x for x in character_names if x[1] > character_threshold]
            character_res = dict(character_res)
            sorted_character = sorted(character_res.items(), key=lambda x: x[1], reverse=True)
            
            if self.debug_mode:
                logger.debug(f"Processing character tags (threshold {character_threshold}):")
            for tag, confidence in sorted_character:
                if tag not in combined_tags:
                    combined_tags.append(tag)
                    if self.debug_mode:
                        logger.debug(f"  {tag}: {confidence:.3f}")
            
            # 4.4 Process General Tags
            general_names = [labels[i] for i in self.general_indexes]
            general_res = [x for x in general_names if x[1] > general_threshold]
            general_res = dict(general_res)
            sorted_general = sorted(general_res.items(), key=lambda x: x[1], reverse=True)
            
            if self.debug_mode:
                logger.debug(f"Processing general tags (threshold {general_threshold}):")
            for tag, confidence in sorted_general:
                if tag not in combined_tags:
                    combined_tags.append(tag)
                    if self.debug_mode:
                        logger.debug(f"  {tag}: {confidence:.3f}")
            
            # 5. Tag Filtering and Processing
            # 5.1 Remove Undesired Tags
            if undesired_tags:
                before_removal = len(combined_tags)
                combined_tags = [tag for tag in combined_tags if tag not in undesired_tags]
                removed_count = before_removal - len(combined_tags)
                if self.debug_mode:
                    logger.debug(f"Removed {removed_count} undesired tags")
            
            # 5.2 Handle Underscore Removal
            if remove_underscore:
                kaomojis = ["0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", 
                        "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", 
                        "u_u", "x_x", "|_|", "||_||"]
                
                combined_tags = [
                    tag if tag in kaomojis else tag.replace("_", " ")
                    for tag in combined_tags
                ]
                if self.debug_mode:
                    logger.debug("Removed underscores from tags (except kaomojis)")
            
            # 6. Final Caption Generation
            caption = caption_separator.join(combined_tags)
            if self.debug_mode:
                logger.debug(f"Final caption ({len(combined_tags)} tags):")
                logger.debug(caption)
            
            return caption
            
        except Exception as e:
            error_msg = f"Error generating caption for {image_path}: {e}"
            logger.error(error_msg)  # Errors always logged regardless of debug mode
            return "error_generating_caption"

    def prepare_image(self, image_path):
        # Load image
        image = Image.open(image_path).convert('RGBA')
        
        # Create white background
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
        
        # Pad image to square
        max_dim = max(image.size)
        pad_left = (max_dim - image.size[0]) // 2
        pad_top = (max_dim - image.size[1]) // 2
        
        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))
        
        # Resize if needed
        if max_dim != self.target_size:
            padded_image = padded_image.resize(
                (self.target_size, self.target_size),
                Image.BICUBIC
            )
        
        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)
        
        # Convert RGB to BGR
        image_array = image_array[:, :, ::-1]
        
        return np.expand_dims(image_array, axis=0)

    def load_tags(self, tags_df):
        # Store the original names without any processing
        self.tag_names = tags_df["name"].tolist()
        
        # Separate tags by category
        self.rating_indexes = list(np.where(tags_df["category"] == 9)[0])
        self.general_indexes = list(np.where(tags_df["category"] == 0)[0])
        self.character_indexes = list(np.where(tags_df["category"] == 4)[0])
