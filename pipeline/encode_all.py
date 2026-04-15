import os
import gc
import pandas as pd
import torch
from pipeline.config import ENCODER_BATCH_SIZE, ENCODER_MAX_LENGTH, PROCESSED_DATA_PATH, TEXT_FIELDS
from model.encoder import TextEncoder, encode_texts_to_disk


def list_processed_csv_files(processed_data_path):
    files = []
    for file in os.listdir(processed_data_path):
        if file.endswith(".csv"):
            files.append(file.split(".")[0])
    return files


def encode_all(processed_data_path=PROCESSED_DATA_PATH):
    text_fields = list(TEXT_FIELDS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    files = list_processed_csv_files(processed_data_path)
    encoder = TextEncoder(device=device, max_length=ENCODER_MAX_LENGTH)

    for file in files:
        df = pd.read_csv(f"{processed_data_path}/{file}.csv")
        available_text_fields = [field for field in text_fields if field in df.columns]
        if len(available_text_fields) == 0:
            raise ValueError(f"No configured text fields found in {file}.csv")

        for field in available_text_fields:
            df[field] = df[field].fillna("").astype(str)

        texts = df[available_text_fields].agg(" ".join, axis=1).tolist()

        path, dim = encode_texts_to_disk(
            texts=texts,
            encoder=encoder,
            save_path=f"{processed_data_path}/{file}.npy",
            batch_size=ENCODER_BATCH_SIZE
        )

        print(f"Done! Saved to {path}, embedding dim: {dim}")

    encoder.model.to("cpu")
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    encode_all()