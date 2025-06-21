import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



def find_data_pairs(base_dir):
    data_pairs = []
    scenarios_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,f))]
    
    for scenario in scenarios_folders:
        scenario_path = os.path.join(base_dir,scenario)
    
        if "Fault" in scenario:
    
            seismic_files = sorted(glob(os.path.join(scenario_path,"seis*_1_0.npy")))
    
            for seis_files in seismic_files:
                file_basename = os.path.basename(seis_files)
                num = file_basename.split("_")[0].replace("seis","")
                vel_files = os.path.join(scenario_path,f"vel{num}_1_0.npy")
                if os.path.exists(vel_files):
                    data_pairs.append((seis_files,vel_files))
    
        elif "Vel" in scenario or "Style" in scenario:
            data = os.path.join(scenario_path,"data")
            model_dir = os.path.join(scenario_path,"model")

            if os.path.exists(data) and os.path.exists(model_dir):
                data_files = sorted(glob(os.path.join(data,"data*.npy")))
                for data_file in data_files:
                    file_basename = os.path.basename(data_file)
                    num = file_basename.replace("data","").split(".")[0]
                    model_file = os.path.join(model_dir,f"model{num}.npy")
                    if os.path.exists(model_file):
                        data_pairs.append((data_file, model_file))

    return data_pairs
