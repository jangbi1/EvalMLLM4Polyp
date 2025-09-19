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

def calculate_normalized_center(xml_path: str) -> Tuple[Optional[float], Optional[float]]:
    """Reads bounding box information from an XML file and calculates the normalized center coordinates."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_tag, object_tag = root.find('size'), root.find('object')
        if size_tag is None or object_tag is None: return None, None
        img_width, img_height = int(size_tag.find('width').text), int(size_tag.find('height').text)
        bndbox = object_tag.find('bndbox')
        if bndbox is None: return None, None
        xmin, xmax = int(bndbox.find('xmin').text), int(bndbox.find('xmax').text)
        ymin, ymax = int(bndbox.find('ymin').text), int(bndbox.find('ymax').text)
        return (xmin + xmax) / (2 * img_width), (ymin + ymax) / (2 * img_height)
    except Exception as e:
        print(f"Warning: An issue occurred while processing XML file '{xml_path}': {e}")
        return None, None

def parse_prediction(text: str) -> Tuple[str, Optional[float], Optional[float]]:
    """Parses the model's response text to extract the prediction status and coordinates."""
    text = text.strip()
    if re.search(r"Polyp Found:\s*Yes", text, re.IGNORECASE):
        status = "Yes"
        match = re.search(r"Normalized Center:\s*\[?\s*([\d.]+)\s*\]?,\s*\[?\s*([\d.]+)\s*\]?", text, re.IGNORECASE)
        if match:
            try: return status, float(match.group(1)), float(match.group(2))
            except (ValueError, IndexError): return status, None, None
        return status, None, None
    elif re.search(r"Polyp Found:\s*No", text, re.IGNORECASE):
        return "No", None, None
    else:
        return "Parsing Error", None, None

# --- Main Evaluation Function ---

def run_polyp_evaluation(
    client: openai.OpenAI,
    model_id: str,
    positive_few_shot_folder: str,
    negative_few_shot_folder: str,
    positive_few_shot_xml_folder: str,
    test_data_list: List[Tuple[str, str, Optional[float], Optional[float]]],
    num_few_shot: int,
    output_csv_path: str,
    generation_config: dict,
    seed: int
):
    """Performs few-shot/zero-shot inference on a list of test data and saves the results to a CSV file."""
    random.seed(seed)
    
    positive_examples = [{'path': os.path.join(positive_few_shot_folder, fname), 'cx': cx, 'cy': cy}
                         for fname in os.listdir(positive_few_shot_folder)
                         if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
                         for cx, cy in [calculate_normalized_center(os.path.join(positive_few_shot_xml_folder, os.path.splitext(fname)[0] + '.xml'))] if cx is not None]
    negative_examples = [{'path': os.path.join(negative_few_shot_folder, fname)}
                         for fname in os.listdir(negative_few_shot_folder)
                         if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Few-shot examples loaded (Positive: {len(positive_examples)}, Negative: {len(negative_examples)})")

    system_prompt = (
        "You are an expert AI assistant specialized in analyzing colonoscopy images for colorectal polyps.\n"
        "Carefully examine the provided colonoscopy image.\n"
        "1. Determine if a visually distinct polyp is present. Focus on actual polyps and ignore normal mucosal folds or debris.\n"
        "2. If a polyp IS PRESENT, estimate its normalized center coordinates (x, y) between 0.0 and 1.0.\n"
        "3. Respond ONLY in ONE of the following two exact formats:\n"
        "   - If polyp present: Polyp Found: Yes, Normalized Center: [X], [Y]\n"
        "   - If polyp absent: Polyp Found: No\n"
        "Do NOT include any other reasoning or explanatory text."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    loaded_few_shot_images = []
    if num_few_shot > 0:
        num_pos = min(num_few_shot // 2, len(positive_examples))
        num_neg = min(num_few_shot - num_pos, len(negative_examples))
        few_shot_samples = random.sample(positive_examples, num_pos) + random.sample(negative_examples, num_neg)
        random.shuffle(few_shot_samples)
        
        try:
            for example in few_shot_samples:
                img = PIL.Image.open(example['path'])
                loaded_few_shot_images.append(img)
                base64_image = encode_image_to_base64(img)
                
                user_content = [
                    {"type": "text", "text": "Analyze this example image."},
                    {"type": "image_url", "image_url": {"url": base64_image}}
                ]
                messages.append({"role": "user", "content": user_content})

                answer = (f"Polyp Found: Yes, Normalized Center: [{example['cx']:.4f}], [{example['cy']:.4f}]" if 'cx' in example
                          else "Polyp Found: No")
                messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            print(f"Error: Failed to load or encode few-shot images: {e}")
            for img in loaded_few_shot_images: img.close()
            return

    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['ImagePath', 'TrueStatus', 'TrueCenterX', 'TrueCenterY', 'PredictedStatus', 'PredictedCenterX', 'PredictedCenterY', 'RawResponse'])
            writer.writeheader()
            
            total_tests = len(test_data_list)
            print(f"\nStarting inference for {total_tests} test images...")

            for i, (img_path, true_status, true_cx, true_cy) in enumerate(test_data_list):
                print(f"  ({i+1}/{total_tests}) Processing: {os.path.basename(img_path)} (Ground Truth: {true_status})")
                test_image = None
                try:
                    test_image = PIL.Image.open(img_path)
                    base64_test_image = encode_image_to_base64(test_image)

                    final_user_content = [
                        {"type": "text", "text": "Now, analyze the following new test image based on the instructions and examples."},
                        {"type": "image_url", "image_url": {"url": base64_test_image}}
                    ]
                    current_messages = messages + [{"role": "user", "content": final_user_content}]
                    response = client.chat.completions.create(
                          model=model_id,
                          messages=current_messages
                    )
                    time.sleep(1)
                    
                    raw_response = response.choices[0].message.content
                    pred_status, pred_cx, pred_cy = parse_prediction(raw_response)
                
                except Exception as e:
                    raw_response = f"[Execution Error: {e}]"
                    pred_status, pred_cy, pred_cx = "Error", None, None
                finally:
                    writer.writerow({
                        'ImagePath': os.path.basename(img_path),
                        'TrueStatus': true_status, 'TrueCenterX': f"{true_cx:.4f}" if true_cx is not None else '',
                        'TrueCenterY': f"{true_cy:.4f}" if true_cy is not None else '',
                        'PredictedStatus': pred_status, 'PredictedCenterX': f"{pred_cx:.4f}" if pred_cx is not None else '',
                        'PredictedCenterY': f"{pred_cy:.4f}" if pred_cy is not None else '', 'RawResponse': raw_response.replace('\n', ' ')
                    })
                    if test_image: test_image.close()

        print(f"\n Evaluation complete! Results saved to '{output_csv_path}'.")
    except IOError as e:
        print(f"Fatal Error: Failed to write to CSV file: {e}")
    finally:
        for img in loaded_few_shot_images: img.close()
        print("All few-shot image resources have been released.")

# --- Script Execution Block ---
if __name__ == '__main__':
    # --- 1. Set Paths and Parameters ---
    MODEL_ID = 'gpt-4o' 
    FEW_SHOT_POS_IMG_FOLDER = './data/few_shot_examples/positive/images'
    FEW_SHOT_NEG_IMG_FOLDER = './data/few_shot_examples/negative/images'
    FEW_SHOT_POS_XML_FOLDER = './data/few_shot_examples/positive/annotations'
    TEST_POS_IMG_FOLDER = './data/test_set/positive/images'
    TEST_POS_XML_FOLDER = './data/test_set/positive/annotations'
    TEST_NEG_IMG_FOLDER = './data/test_set/negative/images'
    
    FEW_SHOT_VALUES = [0, 2, 4, 8, 16]
    NUM_REPETITIONS = 3
    SEED_START_VALUE = 42
    
    GENERATION_CONFIG_DICT = {'temperature': 0.1}
    OUTPUT_BASE_DIR = f'./results/task1/{MODEL_ID}'

    # --- 2. Prepare Test Data ---
    print("="*50)
    print(f"Starting evaluation using model '{MODEL_ID}'.")
    print("1. Preparing test data list...")
    
    all_test_data = []
    for fname in sorted(os.listdir(TEST_POS_IMG_FOLDER)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(TEST_POS_IMG_FOLDER, fname)
            xml_path = os.path.join(TEST_POS_XML_FOLDER, os.path.splitext(fname)[0] + '.xml')
            all_test_data.append((img_path, "Yes", *calculate_normalized_center(xml_path)))
    if os.path.isdir(TEST_NEG_IMG_FOLDER):
        for fname in sorted(os.listdir(TEST_NEG_IMG_FOLDER)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_test_data.append((os.path.join(TEST_NEG_IMG_FOLDER, fname), "No", None, None))
    if not all_test_data:
        raise SystemExit("Error: No test data found. Please check folder paths.")
    print(f"Total of {len(all_test_data)} test data items prepared.")
    print("="*50)

    # --- 3. Run Evaluation Loop ---
    for num_shots in FEW_SHOT_VALUES:
        repetitions_for_current_shot = 2 if num_shots == 0 else NUM_REPETITIONS
        
        for i in range(repetitions_for_current_shot):
            run_number = i + 1
            current_seed = SEED_START_VALUE + (FEW_SHOT_VALUES.index(num_shots) * NUM_REPETITIONS) + i
            output_csv_dir = os.path.join(OUTPUT_BASE_DIR, f'shot_{num_shots}')
            os.makedirs(output_csv_dir, exist_ok=True)
            output_csv = os.path.join(output_csv_dir, f'results_run_{run_number}_seed_{current_seed}.csv')
            
            print(f"\n Starting Evaluation (Shots: {num_shots}, Repetition: {run_number}/{repetitions_for_current_shot}, Seed: {current_seed})")
            print("-"*50)
            
            run_polyp_evaluation(
                client=client,
                model_id=MODEL_ID,
                positive_few_shot_folder=FEW_SHOT_POS_IMG_FOLDER,
                negative_few_shot_folder=FEW_SHOT_NEG_IMG_FOLDER,
                positive_few_shot_xml_folder=FEW_SHOT_POS_XML_FOLDER,
                test_data_list=all_test_data,
                num_few_shot=num_shots,
                output_csv_path=output_csv,
                generation_config=GENERATION_CONFIG_DICT,
                seed=current_seed
            )