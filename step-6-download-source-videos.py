import os
import json
import requests
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import logging

# Set up logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def youtube_video_search(api_key, query, num_results=10, page_token=None):
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&key={api_key}&type=video&maxResults={num_results}"
    if page_token:
        url += f"&pageToken={page_token}"
    print(f"Querying YouTube Data API with query: {query}")
    response = requests.get(url)
    return response.json()

def display_video_choices(results, start_index=0):
    video_links = []
    items = results.get('items', [])
    for i, item in enumerate(items[start_index:start_index+10]):
        title = item['snippet']['title']
        channel_title = item['snippet']['channelTitle']
        link = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        print(f"{i+1}. Title: {title}")
        print(f"   Channel: {channel_title}")
        print(f"   Link: {link}")
        video_links.append(link)
    return video_links


def download_video_and_transcript(video_url, output_folder, reviewer_name):
    yt = YouTube(video_url)
    video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    video_path = os.path.join(output_folder, f'{reviewer_name}.mp4')
    video_stream.download(output_path=output_folder, filename=f'{reviewer_name}.mp4')
    
    video_id = video_url.split('=')[1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_path = os.path.join(output_folder, f'{reviewer_name}_transcript.json')
    
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
    
    return video_path, transcript_path

def process_device_reviews(device_name, json_type):
    # Load API keys
    with open('keys/google_custom_search_key.txt', 'r', encoding='utf-8') as f:
        api_key = f.read().strip()
    
    with open('keys/google_cse_id.txt', 'r', encoding='utf-8') as f:
        cse_id = f.read().strip()
    
    # Load JSON data
    json_source = f'Reviews/{device_name}/{device_name}_step_1a_script_{json_type}.json'
    with open(json_source, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    reviewers = data['reviewers_sources']
    
    for reviewer in reviewers:
        if 'youtube' in reviewer['link']:
            query = reviewer['reviewer'] + " " + device_name + " review"
            next_page_token = None
            while True:
                try:
                    results = youtube_video_search(api_key, query, num_results=10, page_token=next_page_token)
                    logging.info(f"Results: {results}")
                    video_links = display_video_choices(results)
                    print("A. Next 10 results")
                    print("B. Skip this reviewer")
                    choice = input("Enter the number of the video you want to choose, or A for next results, or B to skip: ").strip().upper()
                    
                    if choice == 'A':
                        next_page_token = results.get('nextPageToken')
                        if not next_page_token:
                            print("No more results available.")
                            break
                    elif choice == 'B':
                        break
                    else:
                        choice_index = int(choice) - 1
                        chosen_video_link = video_links[choice_index]
                        
                        # Create output folder if it doesn't exist
                        output_folder = os.path.join('Reviews', device_name, 'sources')
                        os.makedirs(output_folder, exist_ok=True)
                        
                        # Download video and transcript
                        download_video_and_transcript(chosen_video_link, output_folder, reviewer['reviewer'])
                        
                        print(f"Downloaded video and transcript for {reviewer['reviewer']}")
                        break
                except Exception as e:
                    logging.info(f"Error processing {reviewer['reviewer']}: {e}")
                    break

def main():
    device_name = input("Enter the device name: ")
    json_type = input("Enter the JSON type (google/openai): ")
    process_device_reviews(device_name, json_type)

if __name__ == "__main__":
    main()
