#from video_gloss_mapping import video_gloss_mapping
import os
import random

def load_gloss_to_video_mapping(file_path):
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            video_id, gloss = line.strip().split(': ')
            if gloss not in mapping:
                mapping[gloss] = []
            mapping[gloss].append(video_id)
    return mapping

#mapping = load_gloss_to_video_mapping('video_gloss_mapping.txt')


def caption_to_video_ids(caption, mapping):
    words = caption.lower().split()
    video_ids = []
    for word in words:
        if word in mapping:
            video_ids.append(random.choice(mapping[word]))
        else:
            video_ids.append(f"<{word}>")
    return video_ids

def create_video_sequence_file(video_ids, output_path):
    with open(output_path, 'w') as f:
        for video_id in video_ids:
            f.write(f"{video_id}\n")
    print(f"Video sequence saved to {output_path}")


"""
#from moviepy.editor import VideoFileClip, concatenate_videoclips
#import os
#import subprocess

def combine_videos_ffmpeg(video_ids, video_folder, output_path):
    input_files = []
    for video_id in video_ids:
        video_path = os.path.join(video_folder, f"{video_id}.mp4")
        if os.path.exists(video_path):
            input_files.append(f"file '{video_path}'")

    if not input_files:
        print("No matching videos found.")
        return        
        
    #write input file
    with open('input.txt', 'w') as f:
        f.write('\n'.join(input_files))


    #construct ffmpeg
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'input.txt',
        '-c', 'copy',
        output_path
    ]   

    try:
        subprocess.run(cmd, check=True)
        print(f"Output saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while combining videos: {e}")
    finally:
        #clean up
        os.remove('input.txt')
"""

def process_caption(caption, mapping, output_path):
    video_ids = caption_to_video_ids(caption, mapping)
    create_video_sequence_file(video_ids, output_path)

    # if not video_ids:
    #     print("No matching videos found for the given caption.")
    #     return

    # combine_videos_ffmpeg(video_ids, video_folder, output_path)

#Load mapping
mapping = load_gloss_to_video_mapping('video_gloss_mapping.txt')


#caption = "I want to read a book and drink water using my computer"
caption="The book rests on the table before the chair"
output_path = "video_sequence.txt"
#video_folder = 'videos'
process_caption(caption, mapping, output_path)