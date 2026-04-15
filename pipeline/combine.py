import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

from pipeline.config import COMBINED_CSV_PATH, COMBINED_NPY_PATH, PROCESSED_DATA_PATH

def combine_data(datasets, output_csv, output_npy):
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(output_csv):
        raise Exception("CSV already exist, remove it first")

    total_records = 0
    emb_dim = None
    
    for ds in datasets:
        npy_shape = np.load(f"{PROCESSED_DATA_PATH}/{ds}.npy", mmap_mode='r').shape
        total_records += npy_shape[0]
        if emb_dim is None:
            emb_dim = npy_shape[1]
            
    print(f"Total records to combine: {total_records}, Embedding dimension: {emb_dim}")
    
    mmap_arr = np.lib.format.open_memmap(
        output_npy, mode="w+", dtype=np.float16, shape=(total_records, emb_dim)
    )
    
    current_global_idx = 0
    write_header = True
    
    for ds in datasets:
        print(f"\n--- Stacking {ds} ---")
        csv_path = f"{PROCESSED_DATA_PATH}/{ds}.csv"
        npy_path = f"{PROCESSED_DATA_PATH}/{ds}.npy"
        
        npy_data = np.load(npy_path, mmap_mode='r')
        dataset_records = len(npy_data)
        
        chunk_size = 50000
        for i in tqdm(range(0, dataset_records, chunk_size), desc="Streaming NPY"):
            chunk_end = min(i + chunk_size, dataset_records)
            mmap_arr[current_global_idx + i:current_global_idx + chunk_end] = npy_data[i:chunk_end]
        
        chunk_start_idx = current_global_idx
        for chunk_df in tqdm(pd.read_csv(csv_path, chunksize=100000), desc="Streaming CSV"):
            chunk_df['npy_index'] = range(chunk_start_idx, chunk_start_idx + len(chunk_df))
            
            chunk_df.to_csv(output_csv, mode='a', index=False, header=write_header)
            
            write_header = False
            chunk_start_idx += len(chunk_df)
            
        current_global_idx += dataset_records
        
    mmap_arr.flush()
    del mmap_arr
    print("\nCombine process completed successfully. Saved stacked datasets.")

if Path(COMBINED_CSV_PATH).exists() or Path(COMBINED_NPY_PATH).exists():
    raise Exception("Give another look at the already saved files at the ouput destination") 

files = []
_files = os.listdir(PROCESSED_DATA_PATH)
for file in _files:
    if file.endswith(".csv"):
        file = file[:-4]
        if Path(f"{PROCESSED_DATA_PATH}/{file}.npy").exists():
            files.append(file)
        else:
            raise Exception("All dataset files must have their own corresponding .npy files first")
del _files

print("Files to pe processed: ", files)

combine_data(
    files,
    output_csv=COMBINED_CSV_PATH,
    output_npy=COMBINED_NPY_PATH
)