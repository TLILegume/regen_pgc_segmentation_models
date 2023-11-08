# Mask utilities
import torch
import cv2
from urllib.request import Request, urlopen
from PIL import Image
from torchvision.io import read_image
from _labelbox_config import LB_API_KEY, DATASET_ID, PROJECT_ID
import labelbox as lb
import ndjson
import numpy as np
import os
import scipy.io
from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from deeplabv3.config import AN_PATH, MASK_CLASSES

ndjson_path = os.path.join(AN_PATH, 'annotations_export.ndjson')

# Connect to client
client = lb.Client(api_key=LB_API_KEY)
print(client.headers)

# Grab annotations
with open(ndjson_path, 'r') as f:
	raw_data = ndjson.load(f)

# Loop through row data and 
for row in raw_data:
	external_id = row['data_row']['external_id']
	print(external_id)
	filepath = os.path.join('./data/annotations', external_id.split('.')[0]+"_mask.mat")
	if os.path.exists(filepath):
		continue
	img_height = row['media_attributes']['height']
	img_width = row['media_attributes']['width']
	mask = np.zeros(shape=(img_height, img_width)).astype(np.uint8)
	project_data = row['projects'][PROJECT_ID]
	if project_data['labels'] != []:
		objects = project_data['labels'][0]['annotations']['objects']
		for obj in objects:
			if obj['annotation_kind'] == 'ImageBoundingBox':
				continue
			category = obj['name']
			url = obj['mask']['url']
			req = Request(url, headers=client.headers)
			image = Image.open(urlopen(req))
			img = np.asarray(image)
			idx = np.where(img > 0)
			mask[idx] = MASK_CLASSES[category]
		scipy.io.savemat(os.path.join('./data/annotations', external_id.split('.')[0]+"_mask.mat"), {'data': mask})
	