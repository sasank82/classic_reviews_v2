import openai
import json
import logging
import google.generativeai as gemini
import pathlib
import textwrap
from IPython.display import display, Markdown
import os

# Set up logging
logging.basicConfig(filename='review_generation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s')

# Adding console handler to logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

def read_system_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_reviewers(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def read_api_key(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def read_google_api_key(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def generate_initial_response(system_prompt, device_name, brand_page_url, reviewers, api_key, max_tokens=4096):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps({
            "device_name": device_name,
            "brand_page_url": brand_page_url,
            "reviewers": reviewers
        })}
    ]
    
    openai.api_key = api_key
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        response_format={"type": "json_object"},
        presence_penalty=0.1
    )
    
    content = response.choices[0].message.content
    logging.info(f"Stop Reason: {response.choices[0].finish_reason}")
    return json.loads(content)

def generate_scene_response(system_prompt, device_name, reviewers_and_scenes, scene, previous_scene_data, api_key, max_tokens=4096):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps({
            "device_name": device_name,
            "reviewers_and_scenes": reviewers_and_scenes,
            "scene": scene,
            "note": "Please provide the response in JSON format."
        })}
    ]

    if previous_scene_data is not None:
        messages.append({"role": "assistant", "content": "```json\n" + json.dumps(previous_scene_data) + "\n```"})

    openai.api_key = api_key
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        response_format={"type": "json_object"},
        presence_penalty=0.1
    )
    
    content = response.choices[0].message.content
    return json.loads(content)

def generate_initial_response_gemini(system_prompt, device_name, brand_page_url, reviewers, api_key):
    gemini.configure(api_key=api_key)
    
    prompt = f"""
    System: {system_prompt}
    User: {{
        "device_name": "{device_name}",
        "brand_page_url": "{brand_page_url}",
        "reviewers": {json.dumps(reviewers)}
    }}
    """
    model = gemini.GenerativeModel(model_name="models/gemini-1.5-pro",
                                   generation_config={"response_mime_type": "application/json"})

    response = model.generate_content(prompt)

    logging.info(f"Response: {response}")
    
    # Extract the actual text content
    content_parts = response.candidates[0].content.parts
    content_text = ''.join(part.text for part in content_parts)  # Combine all parts' text
    logging.info(f"Content Text: {content_text}")
    return json.loads(content_text)

def generate_scene_response_gemini(system_prompt, device_name, reviewers_and_scenes, scene, previous_scene_data, api_key):
    gemini.configure(api_key=api_key)
    
    prompt = f"""
    System: {system_prompt}
    User: {{
        "device_name": "{device_name}",
        "reviewers_and_scenes": {json.dumps(reviewers_and_scenes)},
        "scene": {json.dumps(scene)}
    }}
    """
    
    if previous_scene_data is not None:
        prompt += f"\nAssistant: {json.dumps(previous_scene_data)}"
    
    model = gemini.GenerativeModel(model_name="models/gemini-1.5-pro",
                                   generation_config={"response_mime_type": "application/json"})

    response = model.generate_content(prompt)
    logging.info(f"Response: {response}")
    
    # Extract the actual text content
    content_parts = response.candidates[0].content.parts
    content_text = ''.join(part.text for part in content_parts)  # Combine all parts' text
    
    return json.loads(content_text)

def main():
    # Read the system prompts
    system_prompt_1a = read_system_prompt('System_Prompts/step_1a_system_prompt_v1.txt')
    system_prompt_1b = read_system_prompt('System_Prompts/step_1b_system_prompt_v1.txt')

    # Read the reviewers and API keys from files
    reviewers = read_reviewers('standard_reviewers.txt')
    openai_api_key = read_api_key('keys/openai_key.txt')
    google_api_key = read_google_api_key('keys/gemini_key.txt')

    # Take user input for device name, brand page URL, and choice of API
    device_name = input("Enter the device name: ")
    brand_page_url = input("Enter the brand page URL: ")
    api_choice = input("Choose API (openai/google): ").strip().lower()

    # Step 1a: Generate initial response for reviewers and scenes
    if api_choice == "google":
        initial_response = generate_initial_response_gemini(system_prompt_1a, device_name, brand_page_url, reviewers, google_api_key)
    else:
        initial_response = generate_initial_response(system_prompt_1a, device_name, brand_page_url, reviewers, openai_api_key)

    logging.info(f"Initial Response - Reviewers: {initial_response['reviewers_sources']}")
    logging.info(f"Initial Response - Scenes: {initial_response['scenes']}")

    reviewers_and_scenes = {
        "reviewers_sources": initial_response["reviewers_sources"],
        "scenes": initial_response["scenes"]
    }

    # Create the Reviews folder if it doesn't exist
    if not os.path.exists("Reviews"):
        os.makedirs("Reviews")

    # Create the device name folder inside the Reviews folder if it doesn't exist
    device_folder = os.path.join("Reviews", device_name)
    if not os.path.exists(device_folder):
        os.makedirs(device_folder)

    # Save the final script as a JSON file
    if api_choice == "google":
        file_path = os.path.join(device_folder, f"{device_name}_step_1a_script_google.json")
    else:
        file_path = os.path.join(device_folder, f"{device_name}_step_1a_script_openai.json")

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(reviewers_and_scenes, file, ensure_ascii=False, indent=4)

    # Step 1b: Generate detailed responses for each scene
    final_script = {"device_name": device_name, "scenes": []}

    previous_scene_data = None
    for scene in initial_response["scenes"]:
        scene_name = scene["scene_name"]
        logging.info(f"Generating detailed response for scene: {scene_name}")

        if api_choice == "google":
            detailed_scene = generate_scene_response_gemini(system_prompt_1b, device_name, reviewers_and_scenes, scene, previous_scene_data, google_api_key)
        else:
            detailed_scene = generate_scene_response(system_prompt_1b, device_name, reviewers_and_scenes, scene, previous_scene_data, openai_api_key)

        final_script["scenes"].append(detailed_scene)
        previous_scene_data = detailed_scene

    # Save the final script as a JSON file
    # Create the Reviews folder if it doesn't exist
    if not os.path.exists("Reviews"):
        os.makedirs("Reviews")

    # Create the device name folder inside the Reviews folder if it doesn't exist
    device_folder = os.path.join("Reviews", device_name)
    if not os.path.exists(device_folder):
        os.makedirs(device_folder)

    # Save the final script as a JSON file
    if api_choice == "google":
        file_path = os.path.join(device_folder, f"{device_name}_review_script_google.json")
    else:
        file_path = os.path.join(device_folder, f"{device_name}_review_script_openai.json")

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(final_script, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()