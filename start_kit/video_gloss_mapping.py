import json
import os

with open('WLASL_v0.3.json', 'r') as json_file:
    metadata = json.load(json_file)

video_gloss_mapping = {}

video_folder = 'videos'
video_filenames = os.listdir(video_folder)
#print(metadata)

#Iterate through each entry in JSON
for entry in metadata:
    gloss = entry['gloss']
    for instance in entry['instances']:
        video_id = instance['video_id']
        if f"{video_id}.mp4" in video_filenames:
            video_gloss_mapping[video_id] = gloss



# #populate the mapping
# for instance in metadata:
   
#     video_id = instance.get("video_id")
#     gloss = instance.get("gloss")
#     if video_id and gloss:
#         if f"{video_id}.mp4" in video_filenames:
#             video_gloss_mapping[video_id] = gloss

with open('video_gloss_mapping.txt', 'w') as txt_file:
    for video_id, gloss in video_gloss_mapping.items():
        txt_file.write(f"{video_id}: {gloss}\n")

#print(video_gloss_mapping)