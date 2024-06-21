import moviepy.editor as mpy
import json
import os
import logging
import openai
import random
from PIL import Image, ImageFilter
import numpy as np
import textwrap
from moviepy.audio.fx.all import audio_loop
from moviepy.video.fx.all import fadein, fadeout, resize, crop
from moviepy.video.tools.drawing import color_gradient
from concurrent.futures import ThreadPoolExecutor
import time
import cProfile
import pstats
import re
import cv2
import requests
from pathlib import Path
import sys
import langcodes

# Custom StreamHandler that supports UTF-8
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        self.stream = stream

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            if isinstance(msg, str):
                msg = msg.encode("utf-8", "replace").decode("utf-8", "replace")
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Configure logging to use the UTF-8 handler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[UTF8StreamHandler(sys.stdout)])

# Load OpenAI API key
with open("keys/openai_key.txt", "r") as f:
    openai.api_key = f.read().strip()

def load_api_key(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

google_api_key = load_api_key("keys/google_custom_search_key.txt")
google_cse_id = load_api_key("keys/google_cse_id.txt")

def benchmark(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@benchmark
def transcribe_audio(audio_file_path, user_language, script_segment):
    try:
        # Convert user language to ISO-639-1 code
        language_code = langcodes.find(user_language).language

        with open(audio_file_path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                timestamp_granularities=["segment"],
                response_format='verbose_json',
                language=language_code,  # Use the language code
                prompt=script_segment  # Use the script segment as the prompt
            )
        
        if hasattr(response, 'text') and hasattr(response, 'segments'):
            text = response.text 
            segments = [{'start': seg['start'], 'end': seg['end'], 'text': seg['text']} for seg in response.segments]
            logging.info(f"Text: {text}, Segments: {segments}")
            return text, segments
        else:
            logging.error(f"Unexpected response format: {response}")
            return "", []
    except Exception as e:
        logging.error(f"Error transcribing audio file {audio_file_path}: {e}")
        return "", []

def parse_subscene_response(response_text):
    subscenes = []
    response_text = response_text.strip()  # Remove leading/trailing whitespace

    # Ensure response_text is UTF-8 encoded
    try:
        response_text = response_text.encode('utf-8').decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding response_text to UTF-8: {e}")
        return subscenes

    # Use regex to find the JSON block with "subscenes"
    json_match = re.search(r'(\{\s*"subscenes"\s*:\s*\[\s*{.*}\s*\]\s*\})', response_text, re.DOTALL)
    logging.info(f"JSON block match: {json_match}")

    if json_match:
        json_str = json_match.group(0)
        logging.info(f"Extracted JSON block: {json_str}")

        # Escape newlines AND backslashes 
        json_str = json_str.replace("\\", "\\\\")  # Escape backslashes
        json_str = json_str.replace("\\\n", "\\n")  # Escape the backslash before \n
        json_str = json_str.replace('\\"', '"') #Escape the quotes

        try:
            # Attempt to parse the JSON block
            parsed_data = json.loads(json_str)
            subscenes = parsed_data.get("subscenes", [])
            logging.info(f"Successfully parsed JSON subscenes: {subscenes}")

        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from OpenAI response: {e}")
            logging.error(f"Raw JSON string: {json_str}")

            # Attempt to fix the JSON by ensuring it ends properly
            fixed_json_str = fix_incomplete_json(json_str)
            try:
                parsed_data = json.loads(fixed_json_str)
                subscenes = parsed_data.get("subscenes", [])
                logging.info(f"Successfully parsed fixed JSON subscenes: {subscenes}")
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode even after fixing JSON: {e}")
    else:
        logging.error(f"JSON block not found in OpenAI response: {response_text}")

    return subscenes

def fix_incomplete_json(json_str):
    # Attempt to fix common JSON issues, such as trailing commas or missing brackets
    logging.info(f"Attempting to fix JSON: {json_str}")
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # Remove trailing commas
    if not json_str.endswith(']'):
        json_str += ']'
    if json_str.count('{') != json_str.count('}'):
        json_str += '}' * (json_str.count('{') - json_str.count('}'))

    # Fix misplaced keys like "overlays" outside the main array
    json_str = fix_misplaced_keys(json_str)

    return json_str

def fix_misplaced_keys(json_str):
    # Check for and fix misplaced keys like "overlays" outside the main array
    try:
        json_data = json.loads(json_str)
        if isinstance(json_data, list):
            for subscene in json_data:
                if 'overlays' in subscene:
                    overlays = subscene.pop('overlays')
                    if isinstance(overlays, list):
                        subscene['overlays'] = overlays
        json_str = json.dumps(json_data)
    except json.JSONDecodeError as e:
        logging.error(f"Error in fix_misplaced_keys: {e}")
    return json_str

def call_openai_api(section_json, transcription, time_marks, audio_clip):
    system_prompt = open("System_Prompts/system-prompt-step-5-generating-sub-scenes-v2.txt", "r", encoding='utf-8').read()
    user_message = f"""
    Scene_Info_JSON: {json.dumps(section_json, ensure_ascii=False)}
    Transcription: {transcription}
    Time_Marks_JSON: {json.dumps(time_marks, ensure_ascii=False)}
    Audio Duration: {audio_clip.duration} 
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        subscene_data = response.choices[0].message.content
        return parse_subscene_response(subscene_data)
    except Exception as e:
        logging.error(f"Error during OpenAI API call: {e}")
        return []

def google_custom_search(api_key, cse_id, query, num_results=10):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cse_id}&num={num_results}&searchType=image"
    print(f"Querying Google Custom Search API with query: {query}")
    response = requests.get(url)
    return response.json()

def download_image(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded: {save_path}")
    except Exception as e:
        print(f"Failed to download image from {url}. Error: {e}")

def process_subscene_images(subscene, api_key, cse_id, device_name):
    for i, image_clip in enumerate(subscene.get('image_clips', [])):
        query = image_clip['search_query']
        results = google_custom_search(api_key, cse_id, query)
        if 'items' in results:
            for j, item in enumerate(results['items'][:10]):
                image_url = item['link']
                image_path = Path(f"Reviews/{device_name}/assets/images/{subscene['subscene_label']}_image{i+1}_{j+1}.jpg")
                image_path.parent.mkdir(parents=True, exist_ok=True)
                download_image(image_url, image_path)
                if 'image_paths' not in image_clip:
                    image_clip['image_paths'] = []
                image_clip['image_paths'].append(str(image_path))

def select_images(subscene):
    for image_clip in subscene['image_clips']:
        image_paths = image_clip.get('image_paths', [])
        print(f"Subscene: {subscene['subscene_label']} - Image Clip: {image_clip['search_query']}")
        for idx, path in enumerate(image_paths):
            print(f"{idx+1}: {path}")
        selected_indices = input(f"Select image IDs for the query '{image_clip['search_query']}' (comma-separated): ")
        selected_indices = [int(idx.strip()) - 1 for idx in selected_indices.split(",")]
        selected_images = [image_paths[idx] for idx in selected_indices]
        image_clip['selected_image_paths'] = selected_images

@benchmark
def rescale_and_pad_image(image_path, target_width, target_height):
    try:
        image = Image.open(image_path)
        if image.height > 0 and target_height > 0:
            aspect_ratio = image.width / image.height
            if (target_width / target_height) < aspect_ratio:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)

            image = image.resize((new_width, new_height), Image.LANCZOS)
            padded_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
            padded_image.paste(image, ((target_width - new_width) // 2, (target_height - new_height) // 2))
            return np.array(padded_image)
        else:
            return None
    except Exception as e:
        logging.error(f"Error rescaling and padding image: {e}")
        return None

@benchmark
def create_text_overlay(subscene, device_name, title, text, duration, start_time, text_position, box_position, box_size, section, title_fontsize, text_fontsize, title_color, text_color, max_width=800):
    try:
        if text is None:
            text = ""

        logging.info(f"Creating text overlay with title: {title}, text: {text}")
        logging.info(f"Parameters: title_fontsize={title_fontsize}, text_fontsize={text_fontsize}, title_color={title_color}, text_color={text_color}, box_size={box_size}")
        logging.info(f"Text Position: {text_position} - Box Position: {box_position}")
        logging.info(f"Start Time: {start_time}, Duration: {duration} seconds")  

        try:
            title_clip = mpy.TextClip(
                title,
                fontsize=title_fontsize,
                color="yellow",
                method="caption",
                size=(box_size['width'], None)
            ).set_position('center')
            logging.info(f"Title clip created successfully for Title: {title} with width: {title_clip.w} and height: {title_clip.h}")
        except Exception as e:
            logging.error(f"Error creating title clip: {e}")
            return None
        
        try:
            box_clip_for_title = mpy.ColorClip(size=(min(1280, int(1.2*title_clip.w)), min(720, int(1.5*title_clip.h))), color=(128, 128, 128)).set_opacity(0.7)
            title_box_x = (1280 - box_clip_for_title.w) / 2 if box_clip_for_title.w < 1280 else 0
            box_clip_for_title.set_position((title_box_x, box_position['y']))
            logging.info(f"Box Title clip created successfully for Title: {title} with width: {box_clip_for_title.w} and height: {box_clip_for_title.h}") 
        except Exception as e:
            logging.error(f"Error creating box clip for title: {e}")

        title_clip_concat = mpy.CompositeVideoClip(
            [
                box_clip_for_title.set_position('center'),
                title_clip.set_position('center')
            ],
            size=(box_clip_for_title.w, box_clip_for_title.h)
        )

        try:
            text_clip = mpy.TextClip(
                text,
                fontsize=text_fontsize,
                color="white",
                method="caption",
                size=(box_size['width'], None),
            ).set_position('center')
            logging.info(f"Text clip created successfully for Title: {title} with width: {text_clip.w} and height: {text_clip.h}")
            logging.info(f"Text clip size: {text_clip.size}")
        except Exception as e:
            logging.error(f"Error creating text clip: {e}")
            return None

        try:
            box_clip_for_text = mpy.ColorClip(size=(min(1280, int(1.2*text_clip.w)), min(720, int(1.5*text_clip.h))), color=(255, 0, 0)).set_opacity(0.7)
            text_box_x = (1280 - box_clip_for_text.w) / 2 if box_clip_for_text.w < 1280 else 0
            vertical_gap = 10  # Adjust as needed
            text_box_y = box_position['y'] + box_clip_for_title.h + vertical_gap
            box_clip_for_text.set_position((text_box_x, text_box_y))
            logging.info(f"Box Text clip created successfully for Text: {text} with width: {box_clip_for_text.w} and height: {box_clip_for_text.h}")
        except Exception as e:
            logging.error(f"Error creating box clip for text: {e}")
            return None
        
        text_clip_concat = mpy.CompositeVideoClip(
            [
                box_clip_for_text.set_position('center'),
                text_clip.set_position('center')
            ],
            size=(box_clip_for_text.w, box_clip_for_text.h)
        )

        # Calculate overlay size and position
        overlay_width = max(title_clip_concat.w, text_clip_concat.w)
        overlay_height = title_clip_concat.h + text_clip_concat.h + vertical_gap
        overlay_x = (1280 - overlay_width) / 2
        overlay_y = box_position['y']

        # Create overlay
        try:
            overlay = mpy.CompositeVideoClip(
                [title_clip_concat.set_position(('center', 0)), text_clip_concat.set_position(('center', title_clip_concat.h + vertical_gap))],
                size=(overlay_width, overlay_height)
            ).set_position((overlay_x, overlay_y))
            logging.info(f"Overlay created successfully for Title: {title} with width: {overlay.w} and height: {overlay.h} at position: ({overlay_x}, {overlay_y})")
        except Exception as e:
            logging.error(f"Error creating composite video clip: {e}")
            return None
        
        # Save a frame of the overlay for verification
        overlay_dir = f"Reviews/{device_name}/assets/overlays"
        os.makedirs(overlay_dir, exist_ok=True)  # Create the directory if it doesn't exist
        overlay.save_frame(f"{overlay_dir}/{title}_overlay.png", t=0.5) 
        return overlay
    except Exception as e:
        logging.error(f"Error creating text overlay: {e}")
        return None

@benchmark
def create_subscene(subscene, section, audio_clip, device_name):
    try:
        logging.info(f"Creating subscene: {subscene}")
        subscene_duration = subscene['end_time'] - subscene['start_time']
        subscene_end_time = subscene['start_time'] + subscene_duration
        canvas = mpy.ColorClip(size=(1280, 720), color=(128, 128, 128)).set_duration(subscene_duration)

        # Process and Adjust Image Clips
        clips_image = []
        sorted_image_clips = sorted(subscene['image_clips'], key=lambda x: x['start_time'])
        last_end_time = 0
        for image_clip in sorted_image_clips:
            for image_path in image_clip['selected_image_paths']:
                try:
                    image_position = subscene['layout'].get('image_position', {'x': 0, 'y': 0})
                    overlay_position = subscene['layout'].get('overlay_position', {'x': 0, 'y': 0})
                    layout_type = subscene['layout']['type']

                    desired_image_width = overlay_position['x'] if layout_type == "image_with_side_text" else 1280
                    original_image = Image.open(image_path)
                    aspect_ratio = original_image.width / original_image.height
                    target_height = int(desired_image_width / aspect_ratio)

                    image = rescale_and_pad_image(image_path, desired_image_width, target_height)
                    if image is None:
                        logging.error(f"Error processing image {image_path}. Skipping this image.")
                        continue

                    image_boundaries = (
                        image_position['x'],
                        image_position['y'],
                        image_position['x'] + desired_image_width,
                        image_position['y'] + target_height
                    )

                    if image_boundaries[2] > 1280:
                        image_position['x'] = max(0, 1280 - desired_image_width)
                    if image_boundaries[3] > 720:
                        image_position['y'] = max(0, 720 - target_height)

                    image_clip_start_time = max(last_end_time, image_clip['start_time'] - subscene['start_time'])
                    image_clip_end_time = image_clip_start_time + image_clip['duration']
                    if image_clip_end_time > subscene_duration:
                        image_clip['duration'] = subscene_duration - image_clip_start_time

                    clip = mpy.ImageClip(image).set_duration(image_clip['duration']).set_start(image_clip_start_time).set_position((image_position['x'], image_position['y']))
                    clips_image.append(clip)

                    # Save a frame of the overlay for verification                         
                    overlay_dir = f"Reviews/{device_name}/assets/image-clips"                         
                    os.makedirs(overlay_dir, exist_ok=True)  # Create the directory if it doesn't exist                         
                    clip.save_frame(f"{overlay_dir}/overlay_{random.randint(1, 100000)}.png", t=1)

                    logging.info(f"Image '{image_path}' successfully added to subscene.")
                    last_end_time = image_clip_start_time + image_clip['duration']

                except Exception as e:
                    logging.error(f"Error adding image {image_path}: {e}")
                    continue

        # Create Image Composite Video
        image_composite_clip = mpy.CompositeVideoClip(clips_image, size=(1280, 720)).set_fps(24)

        # Process and Adjust Overlay Clips
        clips_overlay = []
        sorted_overlays = sorted(subscene['overlays'], key=lambda x: x['start_time'])
        last_end_time = 0
        for overlay in sorted_overlays:
            try:
                text_position = subscene['layout'].get('text_position', {'x': 0, 'y': 0})
                box_position = subscene['layout'].get('overlay_position', {'x': 0, 'y': 0})
                box_size = subscene['layout'].get('overlay_box_size', {'width': 0, 'height': 0})

                logging.info(f"Creating overlay '{overlay['title']}' with text: {overlay['text']}")

                # Convert absolute start time to relative start time
                overlay_start_time = max(last_end_time, overlay['start_time'] - subscene['start_time'])
                overlay_end_time = overlay_start_time + overlay['duration']
                
                # Adjust duration if overlay end time exceeds subscene end time
                if overlay_end_time > subscene_duration:
                    overlay['duration'] = subscene_duration - overlay_start_time
                    overlay_end_time = overlay_start_time + overlay['duration']

                overlay_clip = create_text_overlay(
                    subscene,
                    device_name,
                    overlay['title'],
                    overlay['text'],
                    overlay['duration'],
                    overlay_start_time,
                    text_position,
                    box_position,
                    box_size,
                    section,
                    title_fontsize=overlay["title_fontsize"],
                    text_fontsize=overlay["text_fontsize"],
                    title_color=overlay["title_color"],
                    text_color=overlay["text_color"]
                )

                if overlay_clip:
                    overlay_clip = overlay_clip.set_start(overlay_start_time).set_duration(overlay['duration'])
                    clips_overlay.append(overlay_clip)
                    logging.info(f"Overlay '{overlay['title']}' successfully added; start_time: {overlay['start_time']} and duration: {overlay['duration']}.")
                    last_end_time = overlay_end_time
                else:
                    logging.info(f"Failed to create overlay clip for '{overlay['title']}'")

            except Exception as e:
                logging.info(f"Error creating overlay '{overlay['title']}': {e}")
                continue

        # Create Overlay Composite Video
        overlay_composite_clip = mpy.CompositeVideoClip(clips_overlay, size=(1280, 720)).set_fps(24)

        # Create Subscene Clip
        subscene_clip = mpy.CompositeVideoClip([image_composite_clip, overlay_composite_clip], size=(1280, 720), use_bgclip=True).set_fps(24).set_start(subscene['start_time']).set_duration(subscene_duration)

        audio_subclip = audio_clip.subclip(subscene['start_time'], subscene['end_time'])
        final_subscene_clip = subscene_clip.set_audio(audio_subclip)
        subscene_dir = f"Reviews/{device_name}/assets/subscenes"
        os.makedirs(subscene_dir, exist_ok=True)
        final_subscene_clip.write_videofile(f"{subscene_dir}/{subscene['subscene_label']}_subscene.mp4", codec="libx264", fps=24)
        return final_subscene_clip

    except Exception as e:
        logging.error(f"Error creating subscene: {e}")
        return None


def process_scene(scene, audio_clip, brand_name, device_name, index, language_code):
    audio_path = os.path.join('Reviews', device_name, language_code, 'audio', f"{scene['scene_name']}_{language_code}.wav")
    if not os.path.exists(audio_path):
        logging.info(f"Audio file not found: {audio_path}")
        return None

    # Construct scene_json based on required structure from review_script
    scene_json = {
        'scene_name': scene['scene_name'],
        'specs': scene.get('specs', []),
        'positives': scene.get('positives', []),
        'negatives': scene.get('negatives', []),
        'quotes': scene.get('quotes', []),
        'user_comments': scene.get('user_comments', []),
        'scores': scene.get('scores', []),
        'voiceover': scene.get('voiceover', "")
    }

    voiceover = ""
    voiceover_path = os.path.join('Reviews', device_name, language_code, 'voiceovers', f"{scene['scene_name']}.txt")
    if os.path.exists(voiceover_path):
        with open(voiceover_path, 'r', encoding='utf-8') as f:
            voiceover = f.read()

    logging.info(f"Scene: {scene['scene_name']}, Language: {language_code}, Voiceover: {voiceover}")

    # Transcribe Audio (Placeholder function)
    transcription, time_marks = transcribe_audio(audio_path, language_code, voiceover)
    
    if not transcription or not time_marks:
        logging.info(f"Skipping scene '{scene['scene_name']}' due to transcription issue.")
        return None
    else:
        logging.info(f"Transcription: {transcription}, Time Marks: {time_marks}")
        # Write time marks to file
        time_marks_file = os.path.join("Reviews", device_name, language_code, "timestamps_openai", f"{sanitize_scene_name(scene['scene_name'])}_{language_code}_timemarks_openai.json")
        with open(time_marks_file, 'w', encoding='utf-8') as f:
            json.dump(time_marks, f, ensure_ascii=False, indent=4)
        logging.info(f"Time marks written to file: {time_marks_file}")

        # Write transcript to file
        transcript_file = os.path.join("Reviews", device_name, language_code, "timestamps_openai", f"{sanitize_scene_name(scene['scene_name'])}_{language_code}_transcript_openai.txt")
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(transcription)
        logging.info(f"Transcript written to file: {transcript_file}")
    
    # Call to OpenAI API for subscene creation
    subscene_data = call_openai_api(scene_json, transcription, time_marks, audio_clip)
    google_api_key = load_api_key("keys/google_custom_search_key.txt")
    google_cse_id = load_api_key("keys/google_cse_id.txt")

    scene_final_clip = None

    for subscene in subscene_data:
        process_subscene_images(subscene, google_api_key, google_cse_id, device_name)
        select_images(subscene)
        try:
            subscene_clip = create_subscene(subscene, scene, audio_clip, device_name)
        #    subscene_dir = os.path.join("Reviews", device_name, "assets", "subscenes")
        #    os.makedirs(subscene_dir, exist_ok=True)
        #    subscene_clip.write_images_sequence(os.path.join(subscene_dir, f"{subscene['subscene_label']}_subscene_%03d.png"), fps=2)
        except Exception as e:
            logging.error(f"Error creating subscene: {e}")
            subscene_clip = None

        if subscene_clip:
            if scene_final_clip is None:
                scene_final_clip = subscene_clip
            else:
                scene_final_clip = mpy.concatenate_videoclips([scene_final_clip, subscene_clip], method="chain")
    scene_dir = f"Reviews/{device_name}/assets/scenes"
    os.makedirs(scene_dir, exist_ok=True)                
    scene_final_clip.write_videofile(f"{scene_dir}/{scene['scene_name']}_{language_code}.mp4", codec="libx264", fps=24)    
    return scene_final_clip

@benchmark
def load_audio(audio_path):
    try:
        return mpy.AudioFileClip(audio_path)
    except Exception as e:
        logging.error(f"Error loading audio file {audio_path}: {e}")
        return None

def sanitize_scene_name(scene_name):
    # Remove any unnecessary quotation marks from the scene name
    return scene_name.replace('"', '')

def create_video(device_name, language_code, json_source):
    brand_name = "Classic Reviews"
    background_music_path = "La Docerola - Quincas Moreira.mp3"
    
    # Select the correct metadata file based on input
    if json_source == 'google':
        metadata_file = os.path.join('Reviews', device_name, f"{device_name}_review_script_google.json")
    else:
        metadata_file = os.path.join('Reviews', device_name, f"{device_name}_review_script_openai.json")
    
    # Validate metadata file existence
    if not os.path.exists(metadata_file):
        logging.error(f"Metadata file not found: {metadata_file}")
        return

    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        logging.error(f"Error loading metadata: {e}")
        return

    clips = []

    with ThreadPoolExecutor() as executor:
        # Aligning with the directory structure and file names given in steps 1, 2, and 3
        audio_futures = [executor.submit(load_audio, os.path.join('Reviews', device_name, language_code, 'audio', f"{scene['scene_name']}_{language_code}.wav")) for scene in metadata["scenes"]]

        for i, scene in enumerate(metadata["scenes"]):
            logging.info(f"Processing scene: {scene['scene_name']}")

            audio_clip = audio_futures[i].result()
            if not audio_clip:
                continue
            
            try:
                scene_clip = process_scene(scene, audio_clip, brand_name, device_name, i, language_code)
            except Exception as e:
                logging.error(f"Error processing scene '{scene['scene_name']}': {e}")
                scene_clip = None
            if scene_clip:
                scene_transition = mpy.TextClip(
                    scene["scene_name"],
                    fontsize=50,
                    color="black",
                    size=(1920, 1080),
                    bg_color="white"
                ).set_duration(2).set_fps(24)

                clips.append(scene_transition)
                clips.append(scene_clip)

    if clips:
        try:
            final_video = mpy.concatenate_videoclips(clips, method="compose")
            logging.info("Successfully combined all scene clips into final video.")
            
            background_music = mpy.AudioFileClip(background_music_path).volumex(0.1)
            logging.info("Loaded background music.")

            if background_music.duration < final_video.duration:
                background_music = audio_loop(background_music, duration=final_video.duration)

            final_video.audio = mpy.CompositeAudioClip([final_video.audio, background_music.set_duration(final_video.duration)])

            video_folder = os.path.join('Reviews', device_name, language_code, 'video')
            os.makedirs(video_folder, exist_ok=True)
            final_video_path = os.path.join(video_folder, f"{device_name}_{language_code}_video.mp4")
            final_video.write_videofile(
                final_video_path,
                fps=24,
                codec="libx264",
                preset="ultrafast"
            )
            logging.info(f"Final video saved at {final_video_path}")
        except Exception as e:
            logging.error(f"Error saving final video: {e}")
    else:
        logging.info("No valid clips to concatenate. Please check your metadata and assets.")

def main():
    device_name = input("Enter the device name: ")
    language_code = input("Enter the language code (e.g., en-US, es-ES): ")
    json_source = input("Choose JSON source (openai/google): ").strip().lower()
    
    # Verify existing folders and files from steps 1, 2, and 3
    review_folder = os.path.join('Reviews', device_name)
    language_folder = os.path.join(review_folder, language_code)
    
    if not os.path.exists(review_folder):
        logging.error(f"Review folder does not exist: {review_folder}")
        return
    
    if not os.path.exists(language_folder):
        logging.error(f"Language folder does not exist: {language_folder}")
        return
    
    if json_source not in ('openai', 'google'):
        logging.error(f"Invalid JSON source: {json_source}")
        return
    
    try:
        create_video(device_name, language_code, json_source)
    except Exception as e:
        logging.error(f"An error occurred while creating the video: {e}")

if __name__ == "__main__":
    main()
