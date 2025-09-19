import os
import PIL.Image
import random
import csv
import re
import time
from typing import List, Optional, Dict
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

def parse_shape_prediction(text: str, categories: List[str]) -> Optional[str]:
    """Parses one of the given categories from the model's response."""
    text = text.strip()
    for cat in categories:
        if re.search(r'\b' + re.escape(cat) + r'\b', text, re.IGNORECASE):
            return cat
    print(f"Warning: Failed to parse a valid shape. Response: {text}")
    return None


# --- Main Evaluation Function ---

def classify_polyp_shape(
    client: openai.OpenAI,
    model_id: str,
    example_data_list: List[Dict],
    test_data_list: List[Dict],
    shape_categories: List[str],
    num_few_shot: int,
    output_csv_path: str,
    generation_config: dict,
    seed: int
):
    """Classifies the shape of polyps and saves the results to a CSV file."""
    random.seed(seed)
    
    system_prompt = (
        "You are an expert in endoscopic image analysis, specializing in colorectal polyp morphology based on the Paris classification.\n"
        "Your task is to classify the polyp in the image into ONE of the following four categories: Ip, Isp, Is, or IIa.\n\n"
        "- Ip (Pedunculated): A polyp clearly attached via a stalk.\n"
        "- Isp (Sub-pedunculated): A polyp with a partial stalk, between pedunculated and sessile.\n"
        "- Is (Sessile): A polyp with a broad base and no distinct stalk.\n"
        "- IIa (Flat elevated): A lesion that is slightly elevated.\n\n"
        "Respond ONLY with the single, most appropriate category label (e.g., 'Ip' or 'Is'). Do not include any other text, reasoning, or formatting."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    loaded_few_shot_images = []
    
    if num_few_shot > 0:
        few_shot_per_cat = max(1, num_few_shot // len(shape_categories))
        few_shot_examples = []
        examples_by_cat = {cat: [] for cat in shape_categories}
        for ex in example_data_list:
            if ex['shape'] in examples_by_cat:
                examples_by_cat[ex['shape']].append(ex)

        for cat in shape_categories:
            if examples_by_cat[cat]:
                num_to_sample = min(few_shot_per_cat, len(examples_by_cat[cat]))
                few_shot_examples.extend(random.sample(examples_by_cat[cat], num_to_sample))
        
        random.shuffle(few_shot_examples)
        
        try:
            for ex in few_shot_examples:
                img = PIL.Image.open(ex['path'])
                loaded_few_shot_images.append(img)
                base64_image = encode_image_to_base64(img)
                
                messages.append({
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": base64_image}}]
                })
                messages.append({"role": "assistant", "content": ex['shape']})

        except Exception as e:
            print(f"Error: Failed to process few-shot examples: {e}")
            for img in loaded_few_shot_images: img.close()
            return
            
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['ImagePath', 'TrueShape', 'PredictedShape', 'RawResponse'])
            writer.writeheader()
            
            for i, test_item in enumerate(test_data_list):
                print(f"  ({i+1}/{len(test_data_list)}) Processing: {os.path.basename(test_item['path'])}")
                test_image = None
                raw_response = "[No Response]"
                try:
                    test_image = PIL.Image.open(test_item['path'])
                    base64_test_image = encode_image_to_base64(test_image)

                    test_user_content = [{"type": "image_url", "image_url": {"url": base64_test_image}}]
                    current_messages = messages + [{"role": "user", "content": test_user_content}]

                    response = client.chat.completions.create(
                        model=model_id,
                        messages=current_messages
                    )
                    time.sleep(1) 

                    raw_response = response.choices[0].message.content
                except Exception as e:
                    raw_response = f"[Execution Error: {e}]"
                finally:
                    pred_shape = parse_shape_prediction(raw_response, shape_categories)
                    writer.writerow({
                        'ImagePath': os.path.basename(test_item['path']),
                        'TrueShape': test_item['shape'],
                        'PredictedShape': pred_shape if pred_shape else 'Parsing Error',
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
    SHAPE_CATEGORIES = ["Ip", "Is", "Isp", "IIa"]
    
    EXAMPLE_PARENT_FOLDER = './data/few_shot_examples/paris_classification'
    TEST_PARENT_FOLDER = './data/test_set/paris_classification'

    FEW_SHOT_VALUES = [0, 4, 8, 16]
    NUM_REPETITIONS = 3
    SEED_START_VALUE = 42
    GENERATION_CONFIG_DICT = {'temperature': 0.1}
    OUTPUT_BASE_DIR = f'./results/task3/{MODEL_ID}'
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # --- 2. Prepare Data ---
    print("="*50)
    print("Preparing data lists...")
    
    example_data = []
    for cat in SHAPE_CATEGORIES:
        cat_folder = os.path.join(EXAMPLE_PARENT_FOLDER, cat)
        if not os.path.isdir(cat_folder): continue
        for fname in sorted(os.listdir(cat_folder)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                example_data.append({'path': os.path.join(cat_folder, fname), 'shape': cat})

    test_data = []
    for cat in SHAPE_CATEGORIES:
        cat_folder = os.path.join(TEST_PARENT_FOLDER, cat)
        if not os.path.isdir(cat_folder):
            print(f"Warning: Test data folder not found: {cat_folder}")
            continue
        for fname in sorted(os.listdir(cat_folder)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_data.append({'path': os.path.join(cat_folder, fname), 'shape': cat})

    if not example_data or not test_data:
        raise SystemExit("Error: Failed to load example or test data.")
    print(f"{len(example_data)} example data and {len(test_data)} test data items prepared.")
    print("="*50)
    
    # --- 3. Run Evaluation ---
    
    for num_shots in FEW_SHOT_VALUES:
        for i in range(NUM_REPETITIONS):
            run_number = i + 1
            current_seed = SEED_START_VALUE + (FEW_SHOT_VALUES.index(num_shots) * NUM_REPETITIONS) + i
            output_csv = os.path.join(OUTPUT_BASE_DIR, f'shape_results_{MODEL_ID}_seed_{current_seed}_run_{run_number}_shot_{num_shots}.csv')
            
            print(f"\n Starting Evaluation (Shots: {num_shots}, Repetition: {run_number}/{NUM_REPETITIONS}, Seed: {current_seed})")
            
            classify_polyp_shape(
                client=client,
                model_id=MODEL_ID,
                example_data_list=example_data,
                test_data_list=test_data,
                shape_categories=SHAPE_CATEGORIES,
                num_few_shot=num_shots,
                output_csv_path=output_csv,
                generation_config=GENERATION_CONFIG_DICT,
                seed=current_seed
            )