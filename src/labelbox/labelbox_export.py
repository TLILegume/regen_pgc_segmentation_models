import labelbox as lb
from _labelbox_config import LB_API_KEY, PROJECT_ID
import json
import os
import pandas as pd
import sys

import ndjson
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from deeplabv3.config import AN_PATH

ndjson_output = os.path.join(AN_PATH, 'annotations_export.ndjson')

client = lb.Client(api_key = LB_API_KEY)
project = client.get_project(PROJECT_ID)


params = {
	"data_row_details": True,
	"attachments": False,
	"project_details": True,
	"performance_details": False,
	"label_details": True,
	"interpolated_frames": False
}

export_task = project.export_v2(
	params=params
	)
export_task.wait_till_done()

if export_task.errors:
  print(export_task.errors)

export_json = export_task.result

with open(ndjson_output, 'w') as f:
	for i in export_json:
		
		f.write(f"{json.dumps(i)}\n")

print('Annotations exported!')