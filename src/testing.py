import pandas as pd

def test_clean():
    yt_videos = pd.read_csv("../yt_videos.csv")
    chosen_video = yt_videos.iloc[0]
    print(chosen_video.head())



test_clean()