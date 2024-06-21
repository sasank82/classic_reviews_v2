import json
import os
import re
import openai
import google.generativeai as gemini
import logging

# Set up logging
logging.basicConfig(filename='review_generation_step2.log', level=logging.INFO,
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

def read_languages(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def read_api_key(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_text(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(data)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def translate_text(system_prompt, text, target_language, api_choice, api_key):
    if api_choice == "google":
        return translate_with_gemini(system_prompt, text, target_language, api_key)
    else:
        return translate_with_openai(system_prompt, text, target_language, api_key)

def translate_with_openai(system_prompt, text, target_language, api_key):
    openai.api_key = api_key
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please translate the following text to {target_language} without waiting for my confirmation.\n\n{text}"}
            ],
            max_tokens=4096,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        translated_text = response.choices[0].message.content
        logging.info(f"Translation (OpenAI) response: {translated_text[:100]}...")
        return translated_text
    except openai.error.OpenAIError as e:
        logging.error(f"Error translating text with OpenAI: {e}")
        return None

def translate_with_gemini(system_prompt, text, target_language, api_key):
    gemini.configure(api_key=api_key)
    prompt = f"{system_prompt}\n\nPlease translate the following text to {target_language} without waiting for my confirmation.\n\n{text}"
    model = gemini.GenerativeModel(model_name="models/gemini-1.5-pro")
    try:
        response = model.generate_content(prompt)
        logging.info(f"Translation (Gemini) response: {response}")
        content_parts = response.candidates[0].content.parts
        translated_text = ''.join(part.text for part in content_parts)
        logging.info(f"Translation (Gemini) response: {translated_text[:100]}...")
        return translated_text
    except Exception as e:
        logging.error(f"Error translating text with Gemini: {e}")
        return None

def generate_ssml(system_prompt, text, api_choice, api_key):
    if api_choice == "google":
        return generate_ssml_with_gemini(system_prompt, text, api_key)
    else:
        return generate_ssml_with_openai(system_prompt, text, api_key)

def generate_ssml_with_openai(system_prompt, text, api_key):
    openai.api_key = api_key
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please convert the following plain text script into SSML without waiting for my confirmation.\n\n{text}"}
            ],
            max_tokens=4096,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        ssml_text = response.choices[0].message.content
        logging.info(f"SSML (OpenAI) response: {ssml_text[:100]}...")
        return ssml_text
    except openai.error.OpenAIError as e:
        logging.error(f"Error generating SSML with OpenAI: {e}")
        return None

def generate_ssml_with_gemini(system_prompt, text, api_key):
    gemini.configure(api_key=api_key)
    prompt = f"{system_prompt}\n\nPlease convert the following plain text script into SSML without waiting for my confirmation.\n\n{text}"
    model = gemini.GenerativeModel(model_name="models/gemini-1.5-pro")
    try:
        response = model.generate_content(prompt)
        logging.info(f"SSML (Gemini) response: {response}")
        content_parts = response.candidates[0].content.parts
        ssml_text = ''.join(part.text for part in content_parts)
        logging.info(f"SSML (Gemini) response: {ssml_text[:100]}...")
        return ssml_text
    except Exception as e:
        logging.error(f"Error generating SSML with Gemini: {e}")
        return None

def main():
    device_name = input("Enter the device name: ")
    json_source = input("Choose JSON source (openai/google): ").strip().lower()
    translation_api_choice = input("Choose API for translation (openai/google): ").strip().lower()
    ssml_api_choice = input("Choose API for SSML encoding (openai/google): ").strip().lower()

    if json_source == "google":
        json_file = os.path.join('Reviews', device_name, f'{device_name}_review_script_google.json')
    else:
        json_file = os.path.join('Reviews', device_name, f'{device_name}_review_script_openai.json')

    if not os.path.exists(json_file):
        logging.error(f"Error: JSON file '{json_file}' does not exist.")
        return

    # Load JSON data
    data = load_json(json_file)
    scenes = data.get("scenes", [])

    # Read system prompts
    translate_prompt = read_system_prompt('System_Prompts/step_2a_translate_prompt.txt')
    encode_prompt = read_system_prompt('System_Prompts/step_2b_encode_prompt.txt')

    # Read API keys
    openai_api_key = read_api_key('keys/openai_key.txt')
    google_api_key = read_api_key('keys/gemini_key.txt')

    # Read target languages
    languages = read_languages('languages.txt')

    # Folder management
    reviews_folder = os.path.join("Reviews", device_name)
    if not os.path.exists(reviews_folder):
        os.makedirs(reviews_folder)

    for language in languages:
        language_folder = os.path.join(reviews_folder, language)
        voiceovers_folder = os.path.join(language_folder, 'voiceovers')
        ssml_folder = os.path.join(language_folder, 'ssml_files')

        os.makedirs(voiceovers_folder, exist_ok=True)
        os.makedirs(ssml_folder, exist_ok=True)

        for scene in scenes:
            scene_name = scene["scene_name"]
            voiceover_text = scene["voiceover"]

            # Translate text
            if language.lower() != "english":
                logging.info(f"Translating scene '{scene_name}' to {language}...")
                translated_text = translate_text(translate_prompt, voiceover_text, language, translation_api_choice, openai_api_key if translation_api_choice == "openai" else google_api_key)
                if not translated_text:
                    logging.error(f"Failed to translate scene '{scene_name}' to {language}.")
                    continue
                translated_file_path = os.path.join(voiceovers_folder, f'{scene_name}.txt')
                save_text(translated_text, translated_file_path)
            else:
                translated_text = voiceover_text

            # Generate SSML
            logging.info(f"Generating SSML for scene '{scene_name}' in {language}...")
            ssml_text = generate_ssml(encode_prompt, translated_text, ssml_api_choice, openai_api_key if ssml_api_choice == "openai" else google_api_key)
            if not ssml_text:
                logging.error(f"Failed to generate SSML for scene '{scene_name}' in {language}.")
                continue
            ssml_file_path = os.path.join(ssml_folder, f'{scene_name}.ssml')
            save_text(ssml_text, ssml_file_path)

    logging.info("Process completed successfully.")

if __name__ == "__main__":
    main()
