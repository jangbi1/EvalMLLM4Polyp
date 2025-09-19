import os
import PIL.Image
import random
import xml.etree.ElementTree as ET
import csv
import re
import time
from typing import List, Tuple, Optional
import base64
import io

import openai
from dotenv import load_dotenv

# --- Basic Setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Error: Please set the OPENAI_API_KEY environment variable.")

client = openai.OpenAI(api_key=api_key)


# --- Utility Functions ---

def encode_image_to_base64(image: PIL.Image.Image, format="JPEG") -> str:
    """Encodes a PIL image into a Base64 string."""
    buffered = io.BytesIO()
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

def calculate_normalized_center_and_dims(xml_path: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Calculates normalized center coordinates (cx, cy) and dimensions (w, h) from an XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_tag, object_tag = root.find('size'), root.find('object')
        if size_tag is None or object_tag is None: return None, None, None, None
        img_width, img_height = int(size_tag.find('width').text), int(size_tag.find('height').text)
        bndbox = object_tag.find('bndbox')
        if bndbox is None: return None, None, None, None
        xmin, xmax = int(bndbox.find('xmin').text), int(bndbox.find('xmax').text)
        ymin, ymax = int(bndbox.find('ymin').text), int(bndbox.find('ymax').text)

        center_x, center_y = (xmin + xmax) / (2 * img_width), (ymin + ymax) / (2 * img_height)
        width, height = (xmax - xmin) / img_width, (ymax - ymin) / img_height
        return center_x, center_y, width, height
    except Exception as e:
        print(f"Warning: Error processing XML '{xml_path}': {e}")
        return None, None, None, None

def parse_bbox_prediction(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Parses width and height from the model's response text."""
    text = text.strip()
    width_match = re.search(r"width:\s*\[?\s*([\d.]+)\s*\]?", text, re.IGNORECASE)
    height_match = re.search(r"height:\s*\[?\s*([\d.]+)\s*\]?", text, re.IGNORECASE)
    
    pred_w = float(width_match.group(1)) if width_match else None
    pred_h = float(height_match.group(1)) if height_match else None
    
    if pred_w is None or pred_h is None:
        print(f"Warning: Failed to parse width or height. Response: {text}")
    return pred_w, pred_h


# --- Main Evaluation Function ---

def predict_bbox_dimensions(
    client: openai.OpenAI,
    model_id: str,
    example_data_list: List[dict],
    test_data_list: List[dict],
    num_few_shot: int,
    output_csv_path: str,
    seed: int
):
    """Predicts bounding box dimensions and saves the results to a CSV file."""
    random.seed(seed)
    
    system_prompt = (
        "You are an expert AI assistant specialized in analyzing colonoscopy images for colorectal polyps.\n"
        "Your task is to predict the normalized width and height of the bounding box that accurately captures the polyp, "
        "Given an endoscopic image and the normalized center coordinates of the polyp within that image.\n"
        "Normalized coordinates and dimensions are values between 0.0 and 1.0.\n"
        "Respond ONLY in the following format:\n"
        "  Predicted Width: [width], Predicted Height: [height]\n"
        "Do not add any extra text, explanations, or introductions."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    loaded_few_shot_images = []
    
    if num_few_shot > 0:
        if len(example_data_list) < num_few_shot:
            print(f"Error: Requested number of few-shot examples ({num_few_shot}) is greater than available examples ({len(example_data_list)}).")
            return
        few_shot_examples = random.sample(example_data_list, num_few_shot)
        
        try:
            for ex in few_shot_examples:
                img = PIL.Image.open(ex['path'])
                loaded_few_shot_images.append(img)
                base64_image = encode_image_to_base64(img)
                
                user_content = [
                    {"type": "text", "text": f"Analyze this example. Normalized Center: [{ex['cx']:.4f}], [{ex['cy']:.4f}]"},
                    {"type": "image_url", "image_url": {"url": base64_image}}
                ]
                messages.append({"role": "user", "content": user_content})

                answer = f"Predicted Width: [{ex['w']:.4f}], Predicted Height: [{ex['h']:.4f}]"
                messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            print(f"Error: Failed to process few-shot examples: {e}")
            for img in loaded_few_shot_images: img.close()
            return
            
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['ImagePath', 'TrueCenterX', 'TrueCenterY', 'TrueWidth', 'TrueHeight', 'PredictedWidth', 'PredictedHeight', 'RawResponse']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, test_item in enumerate(test_data_list):
                print(f"  ({i+1}/{len(test_data_list)}) Processing: {os.path.basename(test_item['path'])}")
                test_image = None
                raw_response = "[Skipped: Missing True Center Coordinates]"
                pred_w, pred_h = None, None
                
                if test_item['cx'] is not None and test_item['cy'] is not None:
                    try:
                        test_image = PIL.Image.open(test_item['path'])
                        base64_test_image = encode_image_to_base64(test_image)

                        test_user_content = [
                            {"type": "text", "text": f"Now, analyze this test image. Test Normalized Center: [{test_item['cx']:.4f}], [{test_item['cy']:.4f}]"},
                            {"type": "image_url", "image_url": {"url": base64_test_image}}
                        ]
                        
                        current_messages = messages + [{"role": "user", "content": test_user_content}]

                        response = client.chat.completions.create(
                            model=model_id,
                            messages=current_messages
                        )
                        raw_response = response.choices[0].message.content
                        time.sleep(1)

                        pred_w, pred_h = parse_bbox_prediction(raw_response)
                        
                    except Exception as e:
                        raw_response = f"[Execution Error: {e}]"
                
                writer.writerow({
                    'ImagePath': os.path.basename(test_item['path']),
                    'TrueCenterX': f"{test_item.get('cx', ''):.4f}", 'TrueCenterY': f"{test_item.get('cy', ''):.4f}",
                    'TrueWidth': f"{test_item.get('w', ''):.4f}", 'TrueHeight': f"{test_item.get('h', ''):.4f}",
                    'PredictedWidth': f"{pred_w:.4f}" if pred_w is not None else '',
                    'PredictedHeight': f"{pred_h:.4f}" if pred_h is not None else '',
                    'RawResponse': raw_response.replace('\n', ' ')
                })
                if test_image: test_image.close()
            
        print(f"\n Evaluation complete! Results saved to '{output_csv_path}'.")
    except Exception as e:
        print(f"Fatal Error: An error occurred during the process: {e}")
    finally:
        for img in loaded_few_shot_images: img.close()


# --- Script Execution Block ---
if __name__ == '__main__':
    # --- 1. Set Paths and Parameters ---
    MODEL_ID = 'gpt-4o'
    
    EXAMPLE_IMAGE_FOLDER = './data/few_shot_examples/images'
    EXAMPLE_XML_FOLDER = './data/few_shot_examples/annotations'
    TEST_IMAGE_FOLDER = './data/test_set/images'
    TEST_XML_FOLDER = './data/test_set/annotations'

    FEW_SHOT_VALUES = [0, 2, 4, 8, 16] 
    NUM_REPETITIONS = 3
    SEED_START_VALUE = 42
    OUTPUT_BASE_DIR = f'./results/task2/{MODEL_ID}'
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # --- 2. Prepare Data ---
    print("="*50)
    print("Preparing data lists...")
    
    def load_data_from_folders(img_folder, xml_folder):
        data_list = []
        for fname in sorted(os.listdir(img_folder)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                xml_path = os.path.join(xml_folder, os.path.splitext(fname)[0] + '.xml')
                if os.path.exists(xml_path):
                    cx, cy, w, h = calculate_normalized_center_and_dims(xml_path)
                    if all(v is not None for v in [cx, cy, w, h]):
                        data_list.append({'path': os.path.join(img_folder, fname), 'cx': cx, 'cy': cy, 'w': w, 'h': h})
        return data_list

    example_data = load_data_from_folders(EXAMPLE_IMAGE_FOLDER, EXAMPLE_XML_FOLDER)
    test_data = load_data_from_folders(TEST_IMAGE_FOLDER, TEST_XML_FOLDER)
    
    if not example_data or not test_data:
        raise SystemExit("Error: Failed to load example or test data. Check folder paths and content.")
    print(f"{len(example_data)} example data and {len(test_data)} test data items prepared.")
    print("="*50)
    
    # --- 3. Run Evaluation Loop ---
    
    for num_shots in FEW_SHOT_VALUES:
        for i in range(NUM_REPETITIONS):
            run_number = i + 1
            current_seed = SEED_START_VALUE + (FEW_SHOT_VALUES.index(num_shots) * NUM_REPETITIONS) + i
            output_csv = os.path.join(OUTPUT_BASE_DIR, f'bbox_results_{MODEL_ID}_seed_{current_seed}_run_{run_number}_shot_{num_shots}.csv')
            
            print(f"\n Starting Evaluation (Shots: {num_shots}, Repetition: {run_number}/{NUM_REPETITIONS}, Seed: {current_seed})")
            print("-"*50)
            
            predict_bbox_dimensions(
                client=client,
                model_id=MODEL_ID,
                example_data_list=example_data,
                test_data_list=test_data,
                num_few_shot=num_shots,
                output_csv_path=output_csv,
                seed=current_seed
            )