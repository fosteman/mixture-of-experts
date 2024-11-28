import runpod
import json
import os
import logging
import base64
import cv2
import numpy as np
import torch
import uuid
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.color import rgb2lab
import math

# Ensure '/tmp' directory exists
os.makedirs('/tmp', exist_ok=True)

# Set up device and worker configuration
device = 'cpu'
print('CPUs available: ', os.cpu_count())
max_workers = os.cpu_count()
if torch.backends.mps.is_available():
    device = 'mps'
    print("MPS support is RAW and unimplemented by Apple atm of this commit")
elif torch.cuda.is_available():
    device = 'cuda:0'
    print("Using CUDA backend")
else:
    device = 'cpu'
    print("Using CPU backend")

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory for models
models_dir = os.path.join(os.getcwd(), 'models')

# Load models
try:
    object_detection_model_path = os.path.join(models_dir, 'object-detector.nov28.pt')
    object_detection_model = YOLO(object_detection_model_path)
    logging.info(f"Model loaded: {object_detection_model_path} ")
except Exception as e:
    logging.error(f"Failed to load object detection model: {e}")
    object_detection_model = None

try:
    caeca_content_model_path = os.path.join(models_dir, 'caeca.content-segment.nov13.pt')
    caeca_content_model = YOLO(caeca_content_model_path)
    logging.info(f"Model loaded: {caeca_content_model_path} ")
except Exception as e:
    logging.error(f"Failed to load caeca content model: {e}")
    caeca_content_model = None

try:
    gizzard_segmentation_model_path = os.path.join(models_dir, 'gizzard.segment.nov25.pt')
    gizzard_segmentation_model = YOLO(gizzard_segmentation_model_path)
    logging.info(f"Model loaded: {gizzard_segmentation_model_path} ")
except Exception as e:
    logging.error(f"Failed to load gizzard segmentation model: {e}")
    gizzard_segmentation_model = None

# Model cache to avoid redundant loading
model_cache = {}

# Function to load model and use cache if already loaded
def load_model(model_path):
    if model_path not in model_cache:
        try:
            model = YOLO(model_path)
            model_cache[model_path] = model
            logging.info(f"Model loaded: {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model {model_path}: {e}")
            model_cache[model_path] = None
    return model_cache[model_path]

# Mapping of organ features models
classification_models = {
    'gizzard': {
        'blisters': os.path.join(models_dir, 'gizzard.body_base_classic.nov28.pt'),
    },
    'duodenum': {
        'inflammation': os.path.join(models_dir, 'duodenum.inflammation_severity.oct21.pt'),
        'villus_structure': os.path.join(models_dir, 'duodenum.villus_structure.oct21.pt'),
    },
    'midgut': {
        'inflammation': os.path.join(models_dir, 'duodenum.inflammation_severity.oct21.pt'),
        'villus_structure': os.path.join(models_dir, 'duodenum.villus_structure.oct21.pt'),
    },
    'distal-ileum': {
        'inflammation': os.path.join(models_dir, 'duodenum.inflammation_severity.oct21.pt'),
        'villus_structure': os.path.join(models_dir, 'duodenum.villus_structure.oct21.pt'),
    },
    'caeca': {
        'gas': os.path.join(models_dir, 'caeca.gas.nov15.release.pt'),
    },
}

# Load classification models at startup
loaded_classification_models = {}
for organ, features in classification_models.items():
    loaded_classification_models[organ] = {}
    for feature, model_path in features.items():
        model = load_model(model_path)
        if model:
            loaded_classification_models[organ][feature] = model

def handler(job):
    """Handler function to process jobs."""
    try:
        job_input = job['input']
        if 'image' not in job_input:
            return {'error': 'No image provided'}

        image_data = job_input['image']
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        # Convert to numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        # Read image using cv2
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return {'error': 'Failed to decode image'}

        # Save the image to a temporary file
        filename = f"{uuid.uuid4()}.jpg"
        save_path = os.path.join('/tmp', filename)
        cv2.imwrite(save_path, image)
        filepaths = [save_path]

        # Perform object detection
        detected_objects = run_detection(filepaths)

        if detected_objects is None:
            return {'error': 'No objects detected'}

        organ = detected_objects['name'].lower()  # Normalize the name of the organ

        # Perform classification based on detected organ
        classification_results = {}

        if organ == 'caeca':
            # Process caeca images
            masked_image = segment_image(save_path)
            if masked_image is None:
                return {'error': 'Image segmentation failed'}

            # Analyze content color
            caeca_content_analysis = get_caeca_content_color(masked_image)
            classification_results['color'] = caeca_content_analysis

            # Analyze gas content
            gas_analysis = get_caeca_content_gas(masked_image)
            classification_results['gas'] = gas_analysis

        elif organ == 'gizzard':
            # Process gizzard images
            gizzard_results = process_gizzard(save_path)
            if gizzard_results is None:
                return {'error': 'Failed to process gizzard image'}
            classification_results = gizzard_results

        if organ in loaded_classification_models:
            classification_results.update(classify_organ(organ, filepaths))

        return {'detected_organs': detected_objects, 'features': classification_results}
    except Exception as e:
        logging.error(f"Error during detection and classification: {e}")
        return {'error': str(e)}

# Include necessary functions from gai.py
def run_detection(image_paths):
    """Detect relevant objects in images using the object detection model."""
    try:
        if object_detection_model is None:
            logging.error("Object detection model is not loaded.")
            return None

        results = object_detection_model(image_paths, save=False, device=device, stream=False,)
        detections = []
        for r in results:
            detection = json.loads(r.to_json())
#             print(object_detection_model.names)
#             print(detection[0]['class'])
            if detection[0]['class'] in [1,2,3,4,5]:
                detections.extend(detection)

        return detections[0] if detections else None
    except Exception as e:
        logging.error(f"Error detecting relevant objects: {e}")
        return None

def segment_image(image_path):
    """Segment the image using the caeca content model."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to load image: {image_path}")
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        confidences = [0.6, 0.5, 0.4, 0.3]
        for conf in confidences:
            results = caeca_content_model.predict(source=image_path, save=False, stream=False, device=device, conf=conf)
            masks = results[0].masks

            if masks is not None and len(masks.data) > 0:
                mask = masks.data[0].cpu().numpy()
                if mask.shape[:2] != image_rgb.shape[:2]:
                    mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                masked_image = np.where(mask[..., None] > 0, image_rgb, 0)
                return masked_image
            else:
                logging.warning(f"No mask found at confidence {conf}. Trying lower confidence.")

        logging.error("No mask found in segmentation results after trying all confidences.")
        return None
    except Exception as e:
        logging.error(f"Error during image segmentation: {e}")
        return None

def get_caeca_content_color(masked_image):
    """Analyze the caeca content color."""
    try:
        best_color, score, avg_color = get_best_color_class(masked_image)
        return {'top1': best_color, 'real-color-reading': avg_color}
    except Exception as e:
        logging.error(f"Error in caeca content color analysis: {e}")
        return None

def get_caeca_content_gas(masked_image):
    """Analyze the gas content of the caeca."""
    try:
        gas_model_path = classification_models['caeca']['gas']
        gas_model = load_model(gas_model_path)

        if gas_model is None:
            logging.error("Gas model could not be loaded.")
            return None

        # Save masked_image to a temporary file
        temp_filename = f"{uuid.uuid4()}_masked.jpg"
        temp_filepath = os.path.join('/tmp', temp_filename)
        cv2.imwrite(temp_filepath, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

        outputs = gas_model.predict(source=[temp_filepath], save=False, device=device, conf=0.5)
        if not outputs:
            logging.error("No gas classification results returned.")
            return None

        top1_class = outputs[0].names[outputs[0].probs.top1]
        top1_confidence = outputs[0].probs.top1conf.item()

        if top1_class == 'foam':
            top1_class = 'present'

        return {'top1': top1_class, 'top1conf': top1_confidence}
    except Exception as e:
        logging.error(f"Error in gas content analysis: {e}")
        return None

def get_best_color_class(image_array):
    """Determine the best color class using the average green channel."""
    color_distances, avg_color = get_color_distances(image_array)

    if color_distances is None:
        return None, 0, avg_color

    pixels = image_array.reshape(-1, 3)
    non_black_pixels = pixels[~np.all(pixels == [0, 0, 0], axis=1)]
    avg_green = np.mean(non_black_pixels[:, 1])

    threshold = 95
    if avg_green <= threshold:
        return 'healthy', 0, avg_color
    else:
        return 'unhealthy', 0, avg_color

# Define the color range boundaries in RGB
colors_rgb = {
    "Green": np.array([153, 170, 187]), # stable
    "Khaki": np.array([100, 60, 40]), # less than ideal
    "Yellow": np.array([150, 90, 40]), # unstable microflora
    "Pale Yellow": np.array([220, 189, 135]), # unstable microflora
    "Butterscotch": np.array([224, 149, 64])
}

# Convert colors to LAB space for perceptual similarity
colors_lab = {name: rgb2lab([[value / 255.0]])[0][0] for name, value in colors_rgb.items()}

def get_color_distances(image_array):
    """
    Calculate the color distances between the image pixels and predefined colors in LAB color space.
    """
    pixels = image_array.reshape(-1, 3) / 255.0  # Normalize RGB values
    non_black_pixels = pixels[~np.all(pixels == [0, 0, 0], axis=1)]

    if len(non_black_pixels) == 0:
        return None, None

    # Convert to LAB color space
    non_black_pixels_lab = rgb2lab(non_black_pixels[None, :, :])[0]

    color_distances = {}
    for color_name, color_value in colors_lab.items():
        distances = np.linalg.norm(non_black_pixels_lab - color_value, axis=1)
        color_distances[color_name] = distances

    # Compute average RGB of non-black pixels
    avg_rgb = np.mean(non_black_pixels, axis=0)
    avg_rgb_int = tuple(np.round(avg_rgb * 255).astype(int))
    avg_color = 'rgb({},{},{})'.format(*avg_rgb_int)

    return color_distances, avg_color

def get_best_color_class(image_array):
    """
    Determine the best color class using a threshold on the average green channel.
    """
    color_distances, avg_color = get_color_distances(image_array)

    if color_distances is None:
        return None, None

    pixels = image_array.reshape(-1, 3)
    non_black_pixels = pixels[~np.all(pixels == [0, 0, 0], axis=1)]
    avg_green = np.mean(non_black_pixels[:, 1])

    threshold = 95
    print(avg_green, threshold)
    if avg_green <= threshold:
        return 'healthy', 0, avg_color
    else:
        return 'unhealthy', 0, avg_color

def process_gizzard(image_path):
    """Process the gizzard image."""
    try:
        if gizzard_segmentation_model is None:
            logging.error("Gizzard segmentation model is not loaded.")
            return None

        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to load image: {image_path}")
            return None

        original_height, original_width = image.shape[:2]

        results = gizzard_segmentation_model.predict(source=image, device='mps', conf=0.5)

        if not results or not results[0].masks:
            logging.error(f"No masks found for image: {image_path}")
            return None

        masks = results[0].masks
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = gizzard_segmentation_model.names

        region_masks = {}
        lesion_masks = []
        region_centroids = {}

        for i, cls_id in enumerate(classes):
            mask = masks.data[i].cpu().numpy()
            if mask.shape != (original_height, original_width):
                mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.uint8)
            class_name = class_names[cls_id]

            if class_name == "lesion":
                lesion_masks.append(mask)
            else:
                region_masks[class_name] = mask
                moments = cv2.moments(mask)
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    region_centroids[class_name] = (cx, cy)

        lesion_counts = {region: 0 for region in region_masks.keys()}
        for lesion_mask in lesion_masks:
            contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                moments = cv2.moments(contours[0])
                if moments["m00"] > 0:
                    lx = int(moments["m10"] / moments["m00"])
                    ly = int(moments["m01"] / moments["m00"])
                    lesion_centroid = (lx, ly)
                else:
                    continue
            else:
                continue

            closest_region = None
            min_distance = float("inf")

            for region_name, region_centroid in region_centroids.items():
                distance = np.sqrt((region_centroid[0] - lesion_centroid[0]) ** 2 + (region_centroid[1] - lesion_centroid[1]) ** 2)
                region_diagonal = np.sqrt(original_width ** 2 + original_height ** 2)
                if distance <= 0.1 * region_diagonal and distance < min_distance:
                    closest_region = region_name
                    min_distance = distance

            if closest_region:
                lesion_counts[closest_region] += 1

        output = {}
        for region_name in ['base', 'junction', 'body']:
            count = lesion_counts.get(region_name, 0)
            if count == 0:
                output[f"{region_name}_lesions"] = {"top1": "none"}
            else:
                output[f"{region_name}_lesions"] = {"top1": f"{count} lesions"}

        return output

    except Exception as e:
        logging.error(f"Error processing gizzard image: {e}")
        return None

def classify_organ(organ, filepaths):
    """Classify specific aspects of the detected organ."""
    features = {}
    organ_models = loaded_classification_models.get(organ, {})

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_feature = {
            executor.submit(run_classification, model, filepaths): feature
            for feature, model in organ_models.items()
        }

        for future in as_completed(future_to_feature):
            feature = future_to_feature[future]
            try:
                result = future.result()
                features[feature] = result[0]
            except Exception as e:
                logging.error(f"Error running classification for {organ} - {feature}: {e}")
                features[feature] = None

    return features

def run_classification(model, image_paths):
    """Run classification using the specified model."""
    try:
        outputs = model.predict(image_paths, save=False, device=device)
        model_names = outputs[0].names

        return {
            'top1': outputs[0].names[outputs[0].probs.top1],
            'top1conf': outputs[0].probs.top1conf.item(),
            'names': model_names
        },
    except Exception as e:
        logging.error(f"Error running classification: {e}")
        return None, []

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})