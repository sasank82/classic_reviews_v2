import os
import re
import uuid
import logging
import json
from google.cloud import storage, texttospeech_v1beta1 as texttospeech, texttospeech_v1
from google.oauth2 import service_account
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.exceptions import NotFound
import google.api_core.exceptions

# Set up logging
logging.basicConfig(filename='voiceover_generation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

def list_voices(language_code):
    client = texttospeech_v1.TextToSpeechClient()
    logging.info(f"Listing voices for language code: {language_code}")
    try:
        response = client.list_voices(language_code=language_code)
        return response.voices
    except Exception as e:
        logging.error(f"Error listing voices: {e}")
        return None

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    try:
        blob.upload_from_filename(source_file_name)
        logging.info(f'File {source_file_name} uploaded to {destination_blob_name}.')
    except Exception as e:
        logging.error(f"Error uploading file to GCS: {e}")

def delete_from_gcs(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    try:
        if blob.exists():
            blob.delete()
            logging.info(f"Existing file {blob_name} deleted from bucket {bucket_name}.")
        else:
            logging.info(f"No existing file {blob_name} to delete from bucket {bucket_name}.")
    except NotFound as e:
        logging.error(f"File not found: {e}")

def synthesize_long_audio(client, text, output_uri, selected_voice, language_code, local_audio_file):
    text = text.replace('<ssml>', '').replace('</ssml>', '')
    input_text = texttospeech.SynthesisInput(ssml=text)
    voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=selected_voice)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=24000)

    parent = f"projects/classic-reviews-424219/locations/global"

    request = texttospeech.SynthesizeLongAudioRequest(
        parent=parent,
        input=input_text,
        audio_config=audio_config,
        voice=voice,
        output_gcs_uri=output_uri,
    )

    logging.info(f"SynthesizeLongAudioRequest: {request}")

    try:
        operation = client.synthesize_long_audio(request=request)
        logging.info("Waiting for operation to complete...")
        logging.info(f"Operation details: {operation}")

        response = operation.result(timeout=1800)
        logging.info(f"Raw response: {response}")

        # Check if the response contains expected attributes
        if hasattr(response, 'audio_content'):
            logging.info(f"Audio content size: {len(response.audio_content)} bytes")
            with open(local_audio_file, "wb") as f:
                f.write(response.audio_content)
            logging.info(f"Audio content saved to {local_audio_file}")
        else:
            logging.warning("Warning: Audio content is empty. Check the logs for synthesis errors.")

        if hasattr(response, 'error') and response.error:
            logging.error(f"Error in synthesis response: {response.error}")

    except google.api_core.exceptions.GoogleAPICallError as e:
        logging.error(f"Google API call error: {e}")
    except google.api_core.exceptions.RetryError as e:
        logging.error(f"Retry error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    try:
        blob.download_to_filename(destination_file_name)
        logging.info(f"Blob {source_blob_name} downloaded to {destination_file_name}.")
    except Exception as e:
        logging.error(f"Error downloading from GCS: {e}")

def transcribe_audio_with_captions(gcs_uri: str, gcs_output_path: str, language_code: str):
    client = speech_v2.SpeechClient()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[language_code],
        model="long",
        features=cloud_speech.RecognitionFeatures(
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
            enable_word_confidence=True
        ),
    )

    file_metadata = cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)

    request = cloud_speech.BatchRecognizeRequest(
        recognizer="projects/classic-reviews-424219/locations/global/recognizers/classic-reviews",
        config=config,
        files=[file_metadata],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            gcs_output_config=cloud_speech.GcsOutputConfig(
                uri=gcs_output_path,
            ),
        ),
    )

    logging.info(f"BatchRecognizeRequest: {request}")

    try:
        operation = client.batch_recognize(request=request)
        logging.info("Waiting for operation to complete...")
        response = operation.result(timeout=1800)
        logging.info(f"Transcription operation response: {response}")

        file_results = response.results[gcs_uri]
        logging.info(f"Fetching results from {file_results.uri}")

        output_bucket, output_object = re.match(r"gs://([^/]+)/(.*)", file_results.uri).group(1, 2)

        storage_client = storage.Client()
        bucket = storage_client.bucket(output_bucket)
        blob = bucket.blob(output_object)
        results_bytes = blob.download_as_bytes()
        batch_recognize_results = cloud_speech.BatchRecognizeResults.from_json(
            results_bytes, ignore_unknown_fields=True
        )

        full_text = ""
        segments = []

        # Correctly access the words and their start_time
        for result in batch_recognize_results.results:
            for alternative in result.alternatives:
                sentences = re.split(r'(?<=[.!?|।])\s+', alternative.transcript)
                word_offset = 0
                for sentence in sentences:
                    if sentence.strip():
                        full_text += sentence + ' '
                        segment = {
                            "start": alternative.words[word_offset].start_offset.total_seconds(),
                            "end": alternative.words[word_offset + len(sentence.split()) - 1].end_offset.total_seconds(),
                            "text": sentence
                        }
                        segments.append(segment)
                        word_offset += len(sentence.split())


        return full_text.strip(), segments

    except NotFound as e:
        logging.error(f"Error fetching transcription results: {e}")
        return "", []


def main():
    key_path = "keys/Classic_Reviews_Service_Account_Key.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

    bucket_name = "video_audio_uploads"
    device_name = input("Enter the device name: ")

    base_folder = os.path.join("Reviews", device_name)
    if not os.path.exists(base_folder):
        print(f"Error: Directory {base_folder} does not exist.")
        return

    print("Available language folders:")
    language_folders = [folder for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]
    for i, folder in enumerate(language_folders, 1):
        print(f"{i}: {folder}")
    language_folder_index = int(input("Choose a language folder by number: ")) - 1
    language_folder = language_folders[language_folder_index]

    language_folder_path = os.path.join(base_folder, language_folder)

    language_code = input("Enter the language code (e.g., en-US, te-IN): ")
    language_suffix = input("Enter the language suffix (e.g., english, telugu): ")

    ssml_folder_path = os.path.join(language_folder_path, 'ssml_files')
    audio_folder_path = os.path.join(language_folder_path, 'audio')
    timestamps_folder_path = os.path.join(language_folder_path, 'timestamps')
    os.makedirs(audio_folder_path, exist_ok=True)
    os.makedirs(timestamps_folder_path, exist_ok=True)

    credentials = service_account.Credentials.from_service_account_file(key_path)
    client = texttospeech.TextToSpeechLongAudioSynthesizeClient(credentials=credentials)

    voices = list_voices(language_code)
    if not voices:
        print("Error: No voices available.")
        return

    print("Available voices:")
    for i, voice in enumerate(voices, 1):
        print(f"{i}: {voice.name}")

    selected_voice_index = input("Choose a voice by number: ")
    if not selected_voice_index.isdigit():
        print("Error: Invalid input. Please enter a number.")
        return

    selected_voice_index = int(selected_voice_index) - 1
    if selected_voice_index < 0 or selected_voice_index >= len(voices):
        print("Error: Invalid voice selection. Please choose a valid number.")
        return

    selected_voice = voices[selected_voice_index].name
    print(f"Selected voice: {selected_voice}")

    json_source = input("Do you want to use the JSON created using Google or OpenAI? (Enter 'google' or 'openai'): ").strip().lower()
    json_file_path = os.path.join(base_folder, f"{device_name}_review_script_{json_source}.json")

    if not os.path.exists(json_file_path):
        logging.error(f"Error: {json_file_path} does not exist.")
        return

    with open(json_file_path, "r", encoding="utf-8") as f:
        review_data = json.load(f)

    scenes = [scene["scene_name"] for scene in review_data["scenes"]]

    for scene_name in scenes:
        ssml_file = f"{scene_name.replace(' ', '_')}.ssml"
        ssml_file_path = os.path.join(ssml_folder_path, ssml_file)
        if os.path.exists(ssml_file_path):
            with open(ssml_file_path, "r", encoding="utf-8") as f:
                ssml_text = f.read()

            unique_id = str(uuid.uuid4())
            output_gcs_uri = f"gs://{bucket_name}/{device_name.replace(' ', '_')}/{scene_name.replace(' ', '_')}_{language_suffix}_{unique_id}.wav"
            local_audio_file = os.path.join(audio_folder_path, f"{scene_name} {language_suffix}.wav")
            delete_from_gcs(bucket_name, f"{device_name.replace(' ', '_')}/{scene_name.replace(' ', '_')}_{language_suffix}_{unique_id}.wav")

            synthesize_long_audio(client, ssml_text, output_gcs_uri, selected_voice, language_code, local_audio_file)
            download_from_gcs(bucket_name, f"{device_name.replace(' ', '_')}/{scene_name.replace(' ', '_')}_{language_suffix}_{unique_id}.wav", local_audio_file)

            gcs_output_path = f"gs://{bucket_name}/{device_name.replace(' ', '_')}/{scene_name.replace(' ', '_')}_{language_suffix}_{unique_id}.json"
            full_text, segments = transcribe_audio_with_captions(output_gcs_uri, gcs_output_path, language_code)

            # Save transcription results to files
            with open(os.path.join(timestamps_folder_path, f"{scene_name} {language_suffix}_transcript.txt"), "w", encoding="utf-8") as transcript_file:
                transcript_file.write(full_text)

            with open(os.path.join(timestamps_folder_path, f"{scene_name}_{language_suffix}_segments.json"), "w", encoding="utf-8") as segments_file:
                json.dump(segments, segments_file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
