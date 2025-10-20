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
                print(f"✅ Added {len(new_rows)} new videos to CSV.")
            else:
                print("⚠️ No new transcripts added.")
    else:
        print("Video transcript had already been added before.")



#videos_benshapiro = search_videos(youtube = youtube, channel_id = None, max_results = 30, query = "Ben Shapiro climate")
#videos_candaceowens = search_videos(youtube = youtube, channel_id = None, max_results = 30, query = "Candace Owens climate")
#videos_davidpakman = search_videos(youtube = youtube, channel_id = None, max_results = 30, query = "David Pakman Show climate")
#videos_seculartalk = search_videos(youtube = youtube, channel_id = None, max_results = 30, query = "Secular Talk climate")

# user_agent = "ScraperProva 1.0 by Solid_Cupcake5790"
# reddit = praw.Reddit(client_id = client_id, client_secret = client_secret, user_agent = user_agent)

# subreddits = [reddit.subreddit("news"), reddit.subreddit("worldnews"), reddit.subreddit("politics"), reddit.subreddit("thenews"), reddit.subreddit("FOXNEWS")]
# newspapers = {
#     #"nytimes.com": {"name": "NY Times", "stance": "liberal"},
#     "wsj.com": {"name": "Wall Street Journal", "stance": "conservative"},
#     #"theguardian.com": {"name": "The Guardian", "stance": "liberal"},
#     "foxnews.com": {"name": "Fox News", "stance": "conservative"}
# }
# for subreddit in subreddits:
#     for post in subreddit.search(query = "ukraine", sort = "relevance", limit = 1000):
#         post_url = post.url.lower()
#         for domain, info in newspapers.items():
#             if domain in post_url:
#                 print(f"Title: {post.title}\n Journal: {info['name']}\n Stance: {info['stance']}\n Date: {datetime.datetime.fromtimestamp(post.created)}")

# right_wing_subreddits = [reddit.subreddit("Conservative"), reddit.subreddit("Republican")]
# #for subreddit in right_wing_subreddits:

# subreddit = reddit.subreddit("brexit")
# for post in subreddit.search(query="leave", limit = 1000):
#     if post.is_self:
#         if len(post.selftext) < 1000:
#             post.comments.replace_more(limit=0)  # removes MoreComments placeholders
#             all_comments = post.comments.list()[:10]
#             top_comments = sorted(all_comments, key=lambda c: c.score, reverse=True)[:10]

#             print(f"Title: {post.title}")
#             print(f"Text: {post.selftext}")
#             print(f"Date: {datetime.datetime.fromtimestamp(post.created)}")
#             print("----------------------")

#             if not all_comments:
#                 print("No comments found.")
#             else:
#                 for comment in all_comments:
#                     if len(comment.body) > 30 and len(comment.body) < 400:
#                         print(f"Comment Body: {comment.body}")
#                         print(f"Score: {comment.score}")
#                         print("----------------------")
