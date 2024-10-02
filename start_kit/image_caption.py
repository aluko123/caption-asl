
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")


img_location = 'stock-photo.jpg'
raw_img = Image.open(img_location).convert('RGB')

#unconditional image captioning
inputs = processor(raw_img, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
captions = processor.decode(out[0], skip_special_tokens=True)
