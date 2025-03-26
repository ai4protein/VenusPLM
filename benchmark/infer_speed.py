from tqdm import tqdm
import numpy as np
from sortedcontainers import SortedList
from vplm.models.vplm.tokenization_transformer import VPLMTokenizer
from vplm.models.vplm.modeling_transformer import TransformerForMaskedLM
from vplm.models.vplm.configuration_transformer import TransformerConfig
import torch
import numpy as np
from itertools import chain

def pack(lengths, bin_capacity=1024, min_length=10, progress_bar=True):
    bins_remaining = []
    active_bins = SortedList(key=lambda x: x[0])
    items_bin = np.empty(len(lengths), dtype=np.int64)
    num_bins = 0
    for item_idx, length in enumerate(tqdm(lengths, disable=not progress_bar)):
        pos = active_bins.bisect_left((length, -1))
        if pos < len(active_bins):
            remaining, bin_idx = active_bins.pop(pos)
            items_bin[item_idx] = bin_idx
            new_remaining = remaining - length
            bins_remaining[bin_idx] = new_remaining
            if new_remaining >= min_length:
                active_bins.add((new_remaining, bin_idx))
        else:
            bin_idx = num_bins
            num_bins += 1
            items_bin[item_idx] = bin_idx
            remaining = bin_capacity - length
            bins_remaining.append(remaining)
            if remaining >= min_length:
                active_bins.add((remaining, bin_idx))
    return items_bin

@torch.no_grad()
def infer_speed_without_packing(file, model, tokenizer, batch_size=1):
    with open(file, "r") as f:
        texts = f.readlines()
    texts = [text.strip()[:2046] for text in texts]
    
    bins = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    total_time = 0
    total_tokens = 0
    
    for bin in tqdm(bins):
        input_dict = tokenizer(bin, padding=True, return_length=False, return_attention_mask=True, return_tensors="pt")
        input_ids = input_dict["input_ids"].cuda()
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).repeat(input_ids.shape[0], 1).cuda()
        attention_mask = input_dict["attention_mask"].cuda()
        if batch_size == 1:
            attention_mask = None
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
        end_time.record()
        torch.cuda.synchronize()
        total_time += start_time.elapsed_time(end_time)
        total_tokens += input_ids.shape[0] * input_ids.shape[1]
    
    print(f"Total time: {total_time:.2f} ms")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens per second: {total_tokens / (total_time / 1000):.2f}")

@torch.no_grad()
def infer_speed_with_packing(file, model, tokenizer):
    with open(file, "r") as f:
        texts = f.readlines()
    texts = [text.strip()[:2046] for text in texts]
    lengths = [len(text) + 2 for text in texts]
    items_bin = pack(lengths, bin_capacity=2048, min_length=10, progress_bar=True)
    num_bins = np.max(items_bin) + 1
    bins = [[] for _ in range(num_bins)]
    print(f"Packing {len(texts)} sequences into {num_bins} bins")
    for item_idx, bin_idx in enumerate(items_bin):
        bins[bin_idx].append(texts[item_idx])
    
    total_time = 0
    total_tokens = 0
    
    for bin in tqdm(bins):    
        input_dict = tokenizer(bin, padding=False, return_length=True, return_attention_mask=False)
        position_ids = [torch.arange(i) for i in input_dict["length"]]
        position_ids = torch.cat(position_ids, dim=0).cuda().unsqueeze(0)
        input_ids = torch.tensor(list(chain(*input_dict["input_ids"])), dtype=torch.long).cuda().unsqueeze(0)
        lengths = [input_dict["length"], ]
        
        total_tokens += input_ids.shape[0] * input_ids.shape[1]
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(
                input_ids=input_ids, 
                position_ids=position_ids, 
                lengths=lengths
            )
        end_time.record()
        torch.cuda.synchronize()
        total_time += start_time.elapsed_time(end_time)
    
    print(f"Total time: {total_time:.2f} ms")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens per second: {total_tokens / (total_time / 1000):.2f}")
    
            
    
if __name__ == "__main__":
    
    # No packing (SDPA, BATCH-SIZE=32)
    # tokenizer = VPLMTokenizer.from_pretrained("AI4Protein/VenusPLM-300M")
    # config = TransformerConfig.from_pretrained("AI4Protein/VenusPLM-300M", attn_impl="sdpa")
    # model = TransformerForMaskedLM.from_pretrained("AI4Protein/VenusPLM-300M", config=config).eval().cuda().to(torch.bfloat16)
    # infer_speed_without_packing("benchmark/data/swissprot/seqs.txt", model=model, tokenizer=tokenizer, batch_size=32)

    # Packing (FlashAttention)
    config = TransformerConfig.from_pretrained("AI4Protein/VenusPLM-300M", attn_impl="flash_attn")
    tokenizer = VPLMTokenizer.from_pretrained("AI4Protein/VenusPLM-300M")
    model = TransformerForMaskedLM.from_pretrained("AI4Protein/VenusPLM-300M", config=config).eval().cuda().to(torch.bfloat16)
    infer_speed_with_packing("benchmark/data/swissprot/seqs.txt", model=model, tokenizer=tokenizer)