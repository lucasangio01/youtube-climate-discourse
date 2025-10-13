from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi


def get_channel_id(youtube, channel_name):

    request = youtube.channels().list(
        part = "id", 
        forHandle = channel_name
        )
    response = request.execute()
    
    return response["items"][0]["id"]


def search_videos(youtube, channel_id, max_results, query):

    if channel_id:
        request = youtube.search().list(
            part = "snippet",
            channelId = channel_id,
            maxResults = max_results,
            order = "relevance",
            q = query,
            safeSearch = "none",
            type = "video",
            videoDuration = "medium" # between 4 and 20 minutes
        )
        
    else:
            request = youtube.search().list(
            part = "snippet",
            maxResults = max_results,
            order = "relevance",
            q = query,
            safeSearch = "none",
            type = "video",
            videoDuration = "medium" # between 4 and 20 minutes
        )

    response = request.execute()
    videos = []
    for item in response["items"]:
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        videos = [("video_id", video_id), ("title", title), ("url", url)]
        videos.append({
            "title": title,
            "video_id": video_id,
            "url": url,
        })

    return videos


def print_video_transcript(video_id):

    yt_transcript_api = YouTubeTranscriptApi()
    transcript = yt_transcript_api.fetch(video_id = video_id)
    full_text = " ".join([snippet.text for snippet in transcript])
    return full_text

