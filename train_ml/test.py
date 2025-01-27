from PIL import Image
import requests
import torch
import numpy as np
import re
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


device = "cpu"
def crop_and_resize(image, target_size):
    width, height = image.size
    source_size = min(image.size)
    left = width // 2 - source_size // 2
    top = height // 2 - source_size // 2
    right, bottom = left + source_size, top + source_size
    return image.resize(target_size, box=(left, top, right, bottom))

def parse_bbox_and_labels(detokenized_output: str):
  matches = re.finditer(
      '<loc(?P<y0>\d\d\d\d)><loc(?P<x0>\d\d\d\d)><loc(?P<y1>\d\d\d\d)><loc(?P<x1>\d\d\d\d)>'
      ' (?P<label>.+?)( ;|$)',
      detokenized_output,
  )
  labels, boxes = [], []
  fmt = lambda x: float(x) / 1024.0
  for m in matches:
    d = m.groupdict()
    boxes.append([fmt(d['y0']), fmt(d['x0']), fmt(d['y1']), fmt(d['x1'])])
    labels.append(d['label'])
  return np.array(boxes), np.array(labels)

# def display_boxes(image, boxes, labels, target_image_size):
#   h, l = target_size
#   fig, ax = plt.subplots()
#   ax.imshow(image)
#   for i in range(boxes.shape[0]):
#       y, x, y2, x2 = (boxes[i]*h)
#       width = x2 - x
#       height = y2 - y
#       # Create a Rectangle patch
#       rect = patches.Rectangle((x, y),
#                                width,
#                                height,
#                                linewidth=1,
#                                edgecolor='r',
#                                facecolor='none')
#       # Add label
#       plt.text(x, y, labels[i], color='red', fontsize=12)
#       # Add the patch to the Axes
#       ax.add_patch(rect)

#   plt.show()

model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-pt-224").to(device)
processor = AutoProcessor.from_pretrained("google/paligemma2-3b-pt-224")

prompt = "<image>describe en/n"
url = "https://storage.googleapis.com/keras-cv/models/paligemma/cow_beach_1.png"
image = Image.open(requests.get(url, stream=True).raw)


# image = Image.open(
#     "/media/predictions/000.jpg_001.jpg"
# )

image = crop_and_resize(image=image, target_size=(224, 224))
inputs = processor(images=image, text=prompt,  return_tensors="pt").to(device)
inputs = inputs.to(dtype=model.dtype)
# Generate
# generate_ids = model.generate(**inputs, max_length=300)
# processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
prefix_length = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generation = generation[0][prefix_length:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    # print(json.dumps(json.loads(decoded), indent=4))
    
    print(decoded)
