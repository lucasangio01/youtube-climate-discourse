from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import time
import random


def get_channel_id(youtube, channel_name):
    request = youtube.channels().list(part = "id", forHandle = channel_name)
    response = request.execute()
    return response["items"][0]["id"]

def search_videos(youtube, max_results, query):
    request = youtube.search().list(part = "snippet", maxResults = max_results, order = "relevance", q = query, safeSearch = "none", type = "video", videoDuration = "medium") # between 4 and 20 minutes
    response = request.execute()
    videos = []
    for item in response["items"]:
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        videos.append({"title": title, "video_id": video_id, "url": url,})
    return videos

def print_video_transcript(video_id):
    yt_transcript_api = YouTubeTranscriptApi()
    transcript = yt_transcript_api.fetch(video_id = video_id)
    full_text = " ".join([snippet.text for snippet in transcript])
    return full_text

def append_transcript_csv():
    new_rows = []
    temp_title = "Donald Trump's Best Lines on Climate Change"
    temp_id = "qpyFjUsm2PU"
    temp_channel = "Donald Trump"
    time.sleep(random.uniform(2, 5)) # for avoiding to get blocked by YT
    if temp_id not in yt_videos["video_id"].values:
        transcript = print_video_transcript(video_id = temp_id)
        if transcript:  # only add if transcript was successfully fetched
            new_rows.append({"title": temp_title, "video_id": temp_id, "channel": temp_channel, "transcript": transcript})
            if new_rows:
                yt_videos = pd.concat([yt_videos, pd.DataFrame(new_rows)], ignore_index=True)
                yt_videos.to_csv("../yt_videos.csv", index=False)
                print(f"Added {len(new_rows)} new videos to csv")
            else:
                print("No new transcripts added")
    else:
        print("Video transcript had already been added before")
