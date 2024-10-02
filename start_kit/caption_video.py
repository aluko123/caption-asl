#from video_gloss_mapping import video_gloss_mapping
from image_caption import captions
import os
import random
import string

def load_gloss_to_video_mapping(file_path):
    mapping = {}
    #translator = str.maketrans('','', string.punctuation)

    with open(file_path, 'r') as f:
        for line in f:
            video_id, gloss = line.strip().split(': ')
            if gloss not in mapping:
                mapping[gloss] = []
            mapping[gloss].append(video_id)
    return mapping

#mapping = load_gloss_to_video_mapping('video_gloss_mapping.t xt')


def caption_to_video_ids(caption, mapping):
    words = caption.lower().split()
    video_ids = []
    for word in words:
        if word in mapping:
            video_ids.append(random.choice(mapping[word]))
            #video_ids.append(mapping[word])
        else:
            video_ids.append(f"<{word}>")
    return video_ids

def create_video_sequence_file(video_ids, output_path):
    with open(output_path, 'w') as f:
        for video_id in video_ids:
            f.write(f"{video_id}\n")
    print(f"Video sequence saved to {output_path}")


#Notes:
# Video sequence isn;t the best, since the videos are of different resolutions


#using moviepy
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip, ColorClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np



def create_text_clip(text, size, duration, font_size=35, color=(255, 255, 255), bg_color=(0, 0, 0, 128)):
    img = Image.new('RGBA', size, (0, 0, 0, 0))  # create a transparent background
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)

    #Get text size
    left, top, right, bottom = font.getbbox(text)
    text_width, text_height = right - left, bottom - top

    #create a semi-transparent background for subtitles
    bg_height = text_height + 20        # added padding
    bg = Image.new('RGBA', (size[0], bg_height), bg_color)
    img.paste(bg, (0, size[1] - bg_height), bg)

    #make the subtitle show in bottom center
    position = ((size[0] - text_width) // 2, size[1] - text_height - 10)
    draw.text(position, text, font=font, fill=color)
    
    return ImageClip(np.array(img)).set_duration(duration)

def create_video_sequence(video_ids, videos_folder, caption):
    clips = []
    subtitles = []
    current_time = 0
    caption = list(caption.lower().split(" "))
    #print(caption)
    default_size = (320, 240) #default size if there's no matching video

    # get dim from first valid video
    for video_id in video_ids:
        if str(video_id).isdigit():
            video_path = os.path.join(videos_folder, f"{video_id}.mp4")
            if os.path.exists(video_path):
                with VideoFileClip(video_path) as clip:
                    default_size = clip.size
                break

    for  i, video_id in enumerate(video_ids):
        video_id_str = str(video_id)
        if video_id_str.isdigit():
            video_path = os.path.join(videos_folder, f"{video_id_str}.mp4")
            #clip = VideoFileClip(video_path)
            #clips.append(clip)

            # Check if the video file exists
            if os.path.exists(video_path):
                clip = VideoFileClip(video_path)
                clips.append(clip)
                
                #create subtitle for clip
                #print(caption[i])
                subtitle = create_text_clip(caption[i], clip.size, clip.duration)
                #subtitle = TextClip(caption[i], fontsize=24, color='white', font='Arial')
                #subtitle = subtitle.set_position(('center', 'bottom')).set_duration(clip.duration)
                subtitle = subtitle.set_start(current_time)
                subtitles.append(subtitle)

                current_time += clip.duration

            else:
                print(f"Warning: Video file for ID {video_id_str} not found.")
            
        else:
            #non-video words
            #subtitle = TextClip(video_id_str, fontsize=24, color='yellow', font='Arial')
            duration = 1       #1 second duration for non-video words
            color_clip = ColorClip(size=default_size, color=(0, 0, 0)).set_duration(duration)
            clips.append(color_clip)
            #dummy_clip = VideoFileClip(os.path.join(videos_folder, "dummy.mp4"))
            
            subtitle = create_text_clip(video_id_str, default_size, duration, color=(255, 255, 0))
            #subtitle = subtitle.set_position(('center', 'bottom')).set_duration(1)
            subtitle = subtitle.set_start(current_time)
            subtitles.append(subtitle)
            current_time += duration
            print(f"No matching video found for {video_id_str}")
                
    if clips:
        final_clip = concatenate_videoclips(clips, method='compose')
        #final_clip.write_videofile("final_video_sequence.mp4")
        
        #add subtitles
        final_clip_with_subtitles = CompositeVideoClip([final_clip] + subtitles)

        #write final vid
        final_clip_with_subtitles.write_videofile("final_sequence_with_subtitles_3.mp4")

        #close clips
        final_clip_with_subtitles.close()
        final_clip.close()
        for clip in clips:
            clip.close()
        print("Video sequence created successfully.")
    
    else:
        print("No valid video clips found.")

def process_caption(caption, mapping, output_path, video_folder):
    video_ids = caption_to_video_ids(caption, mapping)
    #print(video_ids)
    create_video_sequence_file(video_ids, output_path)
    create_video_sequence(video_ids, video_folder, caption)
    # if not video_ids:
    #     print("No matching videos found for the given caption.")
    #     return

    # combine_videos_ffmpeg(video_ids, video_folder, output_path) 

#Load mapping
mapping = load_gloss_to_video_mapping('video_gloss_mapping.txt')
#print(mapping)

#caption = "I want to read a book and drink water using my computer"

#sentence 1 from paragraph_nlp.txt
#caption="I lived with that woman upstairs four years, and before that time she had tried me indeed. Her character ripened and developed with frightful rapidity. Her vices sprang up fast and rank: they were so strong, only cruelty could check them, and I would not use cruelty"

#sentence 2 from paragraph_nlp.txt
#caption = "Winston had time to learn every detail of her hand. He explored the long fingers, the shapely nails, the work-hardened palm with its row of callouses, the smooth flesh under the wrist. In the same instant it occurred to him that he did not know what colour the girl's eyes were."

caption = captions

caption = caption.translate(str.maketrans('', '', string.punctuation))
output_path = "video_sequence.txt"
video_folder = 'videos'
process_caption(caption, mapping, output_path, video_folder)