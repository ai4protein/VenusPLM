from tqdm import tqdm
import numpy as np
from sortedcontainers import SortedList
    
def sequence_packing(lengths, bin_capacity=1024, min_length=10, progress_bar=True):
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
