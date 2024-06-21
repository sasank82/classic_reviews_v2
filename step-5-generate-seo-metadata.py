import openai
import json
import google.generativeai as gemini
import logging

# Set up logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to read the API key from a file
def read_api_key(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

# Function to read the system prompt from a file
def read_system_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to get keywords and trends using OpenAI
def get_keywords_and_trends_openai(system_prompt, user_message, api_key, max_tokens=4096):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    openai.api_key = api_key
    try:
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
    except Exception as e:
        logging.error(f"Error generating keywords with OpenAI: {e}")
        return None

# Function to get keywords and trends using Google Gemini
def get_keywords_and_trends_google(system_prompt, user_message, api_key):
    gemini.configure(api_key=api_key)
    
    prompt = f"""
    System: {system_prompt}
    User: {user_message}
    """
    model = gemini.GenerativeModel(model_name="models/gemini-1.5-pro",
                                   generation_config={"response_mime_type": "application/json"})

    try:
        response = model.generate_content(prompt)
        logging.info(f"Keywords and Trends (Gemini) response: {response}")
        # Extract the actual text content
        content_parts = response.candidates[0].content.parts
        content_text = ''.join(part.text for part in content_parts)  # Combine all parts' text
        return json.loads(content_text)
    except Exception as e:
        logging.error(f"Error generating keywords with Google Gemini: {e}")
        return None

# Function to generate metadata for YouTube using OpenAI
def generate_metadata_youtube_openai(system_prompt, keywords_trends_json, device_name, language, voiceover_script, api_key, max_tokens=4096):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps({
            "device_name": device_name,
            "language": language,
            "voiceover_script": voiceover_script,
            "keywords_trends": keywords_trends_json
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

# Function to generate metadata for YouTube using Google Gemini
def generate_metadata_youtube_google(system_prompt, keywords_trends_json, device_name, language, voiceover_script, api_key):
    gemini.configure(api_key=api_key)
    
    prompt = f"""
    System: {system_prompt}
    User: {json.dumps({
        "device_name": device_name,
        "language": language,
        "voiceover_script": voiceover_script,
        "keywords_trends": keywords_trends_json
    })}
    """
    model = gemini.GenerativeModel(model_name="models/gemini-1.5-pro",
                                   generation_config={"response_mime_type": "application/json"})

    try:
        response = model.generate_content(prompt)
        logging.info(f"Metadata (Gemini) response: {response}")
        # Extract the actual text content
        content_parts = response.candidates[0].content.parts
        content_text = ''.join(part.text for part in content_parts)  # Combine all parts' text
        return json.loads(content_text)
    except Exception as e:
        logging.error(f"Error generating metadata with Google Gemini: {e}")
        return None

# Function to save the response to a JSON file
def save_response_to_file(response, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(response, ensure_ascii=False, indent=4))

# Function to join voiceovers for each scene
def join_voiceovers(device_name, language, scenes):
    voiceover_folder = f"Reviews/{device_name}/{language}/voiceovers"
    combined_voiceover = ""
    
    for scene in scenes:
        scene_file = f"{voiceover_folder}/{scene['scene_name']}.txt"
        with open(scene_file, 'r', encoding='utf-8') as f:
            combined_voiceover += f.read() + "\n"
    
    return combined_voiceover

# Main function
def main():
    # User inputs
    device_name = input("Enter the device name: ").strip()
    language = input("Enter the language: ").strip()
    
    json_source = input("Do you want to consider OpenAI or Google generated JSON? (Enter 'openai' or 'google'): ").strip().lower()
    keyword_source = input("Do you want to use OpenAI or Google for generating keywords? (Enter 'openai' or 'google'): ").strip().lower()
    metadata_source = input("Do you want to use OpenAI or Google for generating metadata? (Enter 'openai' or 'google'): ").strip().lower()
    
    # Construct the filename based on user inputs
    json_file_path = f"Reviews/{device_name}/{device_name}_review_script_{json_source}.json"
    
    # Read the system prompt
    system_prompt = read_system_prompt('System_Prompts/step_5a_keywords_prompt.txt')
    
    # Create the user message
    user_message = json.dumps({
        "device_name": device_name,
        "language": language
    })
    
    # Generate keywords and trends based on the user choice
    if keyword_source == 'openai':
        api_key = read_api_key('keys/openai_key.txt')
        response = get_keywords_and_trends_openai(system_prompt, user_message, api_key)
    elif keyword_source == 'google':
        api_key = read_api_key('keys/gemini_key.txt')
        response = get_keywords_and_trends_google(system_prompt, user_message, api_key)
    else:
        logging.error("Invalid choice for keyword generation source.")
        return
    
    if response:
        # Save the response to a JSON file
        output_file_path = f"Reviews/{device_name}/{language}/keywords_and_trends_{language}_{keyword_source}.json"
        save_response_to_file(response, output_file_path)
        logging.info(f"Keywords and trends have been saved successfully to {output_file_path}.")
    else:
        logging.error("Failed to generate keywords and trends.")

    # Additional step for generating YouTube metadata
    if response:
        # Read the system prompt for metadata generation
        metadata_system_prompt = read_system_prompt('System_Prompts/step_5b_metadata_prompt.txt')
        
        # Load the list of scenes from the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            review_script_json = json.load(f)
            scenes = review_script_json['scenes']
        
        # Join voiceovers for each scene
        voiceover_script = join_voiceovers(device_name, language, scenes)
        
        # Generate metadata using the selected keywords and trends
        if metadata_source == 'openai':
            api_key = read_api_key('keys/openai_key.txt')
            metadata_response = generate_metadata_youtube_openai(metadata_system_prompt, response, device_name, language, voiceover_script, api_key)
        elif metadata_source == 'google':
            api_key = read_api_key('keys/gemini_key.txt')
            metadata_response = generate_metadata_youtube_google(metadata_system_prompt, response, device_name, language, voiceover_script, api_key)
        else:
            logging.error("Invalid choice for metadata generation source.")
            return
        
        if metadata_response:
            # Save the metadata response to a JSON file
            metadata_output_file_path = f"Reviews/{device_name}/{language}/metadata_{language}_{metadata_source}.json"
            save_response_to_file(metadata_response, metadata_output_file_path)
            logging.info(f"Metadata has been saved successfully to {metadata_output_file_path}.")
        else:
            logging.error("Failed to generate metadata.")

if __name__ == "__main__":
    main()
