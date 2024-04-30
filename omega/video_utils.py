import os
import tempfile
from typing import Optional, BinaryIO, List

import bittensor as bt
import ffmpeg
from pydantic import BaseModel
from yt_dlp import YoutubeDL

from omega.constants import FIVE_MINUTES
import time

from itertools import cycle
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from datasets import load_dataset
dataset = load_dataset("jondurbin/omega-multimodal-ids")
existing_ids = set(dataset["train"]["youtube_id"])

print("existing_ids", len(existing_ids))

from omega.augment import OpenAIAugment
augment = OpenAIAugment(device="cuda:0")

def seconds_to_str(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def clip_video(video_path: str, start: int, end: int) -> Optional[BinaryIO]:
    temp_fileobj = tempfile.NamedTemporaryFile(suffix=".mp4")
    (
        ffmpeg
        .input(video_path, ss=seconds_to_str(start), to=seconds_to_str(end))
        .output(temp_fileobj.name, c="copy")  # copy flag prevents decoding and re-encoding
        .overwrite_output()
        .run(quiet=True)
    )
    return temp_fileobj


def skip_live(info_dict):
    """
    function to skip downloading if it's a live video (yt_dlp doesn't respect the 20 minute 
    download limit for live videos), and we don't want to hang on an hour long stream
    """
    if info_dict.get("is_live"):
        return "Skipping live video"
    return None


class YoutubeResult(BaseModel):
    video_id: str
    title: str
    description: Optional[str]
    length: int
    views: int


def search_videos(query, max_results=8, num_videos=8, filtered_videos=None):
    if filtered_videos is None:
        filtered_videos = []
    
    if len(filtered_videos) >= max_results:
        return filtered_videos[:max_results]

    def search(query):
        try:
            ydl_opts = {
                "format": "worst",
                "dumpjson": True,
                "extract_flat": True,
                "quiet": True,
                "simulate": True,
                "match_filter": skip_live,
                "proxy": "163.172.218.19:17038",
            }

            with YoutubeDL(ydl_opts) as ydl:
                augment_query = augment(query)
                search_query = f"ytsearch{99}:{augment_query}"
                result = ydl.extract_info(search_query, download=False)
                entries = result.get("entries", [])
                
                if entries:
                    videos = [
                        YoutubeResult(
                            video_id=entry["id"],
                            title=entry["title"],
                            description=entry.get("description"),
                            length=(int(entry.get("duration")) if entry.get("duration") else FIVE_MINUTES),
                            views=(entry.get("view_count") if entry.get("view_count") else 0),
                        ) for entry in entries
                    ]

                    new_videos = [video for video in videos if video.video_id not in existing_ids]
                    return new_videos

        except Exception as e:
            bt.logging.warning(f"Error searching for videos: {e}")
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(search, query) for _ in range(3)}
        
        for future in concurrent.futures.as_completed(futures):
            new_videos = future.result()
            if new_videos:
                filtered_videos.extend(new_videos)
                if len(filtered_videos) >= num_videos:
                    break

    return filtered_videos[:max_results]


def get_video_duration(filename: str) -> int:
    metadata = ffmpeg.probe(filename)
    video_stream = next((stream for stream in metadata['streams'] if stream['codec_type'] == 'video'), None)
    duration = int(float(video_stream['duration']))
    return duration


class IPBlockedException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class FakeVideoException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def is_valid_id(youtube_id: str) -> bool:
    return youtube_id is not None and len(youtube_id) == 11


def download_video(
    video_id: str, start: Optional[int]=None, end: Optional[int]=None, proxy: Optional[str]=None
) -> Optional[BinaryIO]:
    if not is_valid_id(video_id):
        raise FakeVideoException(f"Invalid video ID: {video_id}")

    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    temp_fileobj = tempfile.NamedTemporaryFile(suffix=".mp4")
    ydl_opts = {
        "format": "worst",  # Download the worst quality
        "outtmpl": temp_fileobj.name,  # Set the output template to the temporary file"s name
        "overwrites": True,
        "quiet": True,
        "noprogress": True,
        "match_filter": skip_live,
        "proxy": "163.172.218.19:17038",
        # "hls_prefer_native": True,
        # "proxy": "socks5://63i62:k87ohiit@194.76.238.254:5432"
    }

    if start is not None and end is not None:
        ydl_opts["download_ranges"] = lambda _, __: [{"start_time": start, "end_time": end}]

    if proxy is not None:
        ydl_opts["proxy"] = proxy

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Check if the file is empty (download failed)
        if os.stat(temp_fileobj.name).st_size == 0:
            print(f"Error downloading video: {temp_fileobj.name} is empty")
            temp_fileobj.close()
            return None

        return temp_fileobj
    except Exception as e:
        temp_fileobj.close()
        if (
            "Your IP is likely being blocked by Youtube" in str(e) or
            "Requested format is not available" in str(e)
        ):
            raise IPBlockedException(e)
        if any(fake_vid_msg in str(e) for fake_vid_msg in ["Video unavailable", "is not a valid URL", "Incomplete YouTube ID"]):
            raise FakeVideoException(e)
        print(f"Error downloading video: {e}")
        return None


def copy_audio(video_path: str) -> BinaryIO:
    temp_audiofile = tempfile.NamedTemporaryFile(suffix=".aac")
    (
        ffmpeg
        .input(video_path)
        .output(temp_audiofile.name, vn=None, acodec='copy')
        .overwrite_output()
        .run(quiet=True)
    )
    return temp_audiofile
