import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from pipeline.config import ENCODER_MODEL_NAME

class TextEncoder(nn.Module):
    def __init__(self, device, max_length=256):
        super().__init__()

        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_NAME, trust_remote_code=True, dtype="auto")
        self.model = AutoModel.from_pretrained(ENCODER_MODEL_NAME, trust_remote_code=True, dtype="auto").to(device)

        num_paramters = sum([param.numel() for param in self.parameters()])

        print("num_paramters: ", num_paramters)
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, texts: list[str]):
        tokenized_text = self.tokenize(texts)

        embedding_matrix = self.model.get_text_features(**tokenized_text).float()
        return F.normalize(embedding_matrix, dim=1)

    @torch.no_grad()
    def tokenize(self, texts: list[str]):
        safe_texts = [text if (text and text.strip() != "") else " " for text in texts]
        
        tokenized_text = self.tokenizer(safe_texts, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)
        tokenized_text = {key: value.to(self.device) for key, value in tokenized_text.items()}

        return tokenized_text


@torch.no_grad()
def encode_texts_to_disk(texts, encoder, save_path, batch_size=64, dtype=np.float16):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(texts)

    if save_path.exists():
        arr = np.load(save_path, mmap_mode="r")
        if arr.shape[0] == n:
            print(f"Cache exists: {save_path} - skipping.")
            return save_path, int(arr.shape[1])

    safe_texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]

    current_bs = batch_size
    start = 0
    emb_dim = None
    mmap_arr = None

    pbar = tqdm(total=n, desc="Encoding", unit="record", dynamic_ncols=True)

    while start < n:
        end = min(start + current_bs, n)

        text_batch = safe_texts[start:end]

        try:
            batch_emb = encoder(text_batch)
            batch_arr = batch_emb.cpu().numpy().astype(dtype, copy=False)

            if mmap_arr is None:
                emb_dim = int(batch_arr.shape[1])
                mmap_arr = np.lib.format.open_memmap(
                    save_path, mode="w+", dtype=dtype, shape=(n, emb_dim)
                )

            batch_size_actual = end - start
            mmap_arr[start:end] = batch_arr
            start = end

            pbar.update(batch_size_actual)
            pbar.set_postfix({"bs": current_bs, "saved": f"{end}/{n}"})

            del batch_emb, batch_arr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if current_bs == 1:
                raise RuntimeError("OOM at batch_size=1") from exc
            current_bs = max(1, current_bs // 2)
            print(f"\nOOM - reducing batch size to {current_bs}")

    pbar.close()
    if mmap_arr is not None:
        del mmap_arr

    return save_path, emb_dim