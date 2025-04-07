import pandas as pd
import json
import os
from datetime import datetime

# data
ASL_classes = pd.read_json("MS-ASL/MSASL_classes.json").squeeze()
######### Change whether train/test #########
with open("MS-ASL/MSASL_train.json") as f:
    ASL_training_sets = json.load(f)
#############################################

#####################
batch_class_size = 10 # the number of classes for each batch
batch_number = 5 # the number of batches to generate
#####################

# check if batch_class_size and batch_number valid
if (batch_class_size * batch_number) > len(ASL_classes):
    raise ValueError("Number of classes exceeded")

# create folder inside src to store batches from this trial
folder_name = os.path.join('src', f"batch_{batch_class_size}_{batch_number}")
os.makedirs(folder_name, exist_ok=True)

# generate batches
class_name_to_index = {name: idx for idx, name in enumerate(ASL_classes)}
for batch_idx in range(batch_number):
    start = batch_class_size * batch_idx
    end = start + batch_class_size
    batch_classes = ASL_classes[start:end].tolist()
    batch_indices = [class_name_to_index[name] for name in batch_classes]
    
    batch_train_entries = [entry for entry in ASL_training_sets if entry["label"] in batch_indices]
    
    batch= {
        "batch_index": batch_idx,
        "class_names": batch_classes,
        "class_indices": batch_indices,
        "data": batch_train_entries
    }
    
    # store batch_{batch_idx}
    ######### Change whether train/test #########
    out_path = f"{folder_name}/train_batch_{batch['batch_index']}.json"
    #############################################
    with open(out_path, "w") as f:
        json.dump(batch, f, indent=2)
    print(f"Saved batch {batch['batch_index']} to {out_path}")
