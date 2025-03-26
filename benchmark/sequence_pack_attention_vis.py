from tqdm import tqdm
import numpy as np
from sortedcontainers import SortedList
import sys
sys.path.append("/rgzn/limc/VenusPLM/")
from vplm.models.vplm.tokenization_transformer import VPLMTokenizer
from vplm.models.vplm.modeling_transformer import TransformerForMaskedLM
from vplm.models.vplm.configuration_transformer import TransformerConfig
import torch
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as PathEffects
    
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
def show_attention(file, model, tokenizer):
    with open(file, "r") as f:
        texts = f.readlines()
    texts = [text.strip()[:1022] for text in texts]
    lengths = [len(text) + 2 for text in texts]
    items_bin = pack(lengths, bin_capacity=1024, min_length=10, progress_bar=True)
    num_bins = np.max(items_bin) + 1
    bins = [[] for _ in range(num_bins)]
    print(f"Packing {len(texts)} sequences into {num_bins} bins")
    for item_idx, bin_idx in enumerate(items_bin):
        bins[bin_idx].append(texts[item_idx])
    

    for bin in bins:    
        input_dict = tokenizer(bin, padding=False, return_length=True, return_attention_mask=False)
        position_ids = [torch.arange(i) for i in input_dict["length"]]
        position_ids = torch.cat(position_ids, dim=0).cuda().unsqueeze(0)
        input_ids = torch.tensor(list(chain(*input_dict["input_ids"])), dtype=torch.long).cuda().unsqueeze(0)
        attention_mask = [
            (torch.ones(l) + i)    
            for i, l in enumerate(input_dict["length"])
        ]
        attention_mask = torch.cat(attention_mask, dim=0).cuda().unsqueeze(0)
        outputs = model(
            input_ids=input_ids, 
            position_ids=position_ids, 
            attention_mask=attention_mask,
            output_attentions=True
        )
        attention = [torch.mean(attn.squeeze(), dim=0) for attn in outputs.attentions]
        attention = torch.stack(attention, dim=0)
        attention = torch.mean(attention, dim=0)
        attention = attention.cpu().numpy() # [L, L]
        return attention
    
def plot_attention(attention):
    plt.figure(figsize=(12, 10), dpi=300)    
    # Create a custom colormap for a professional look
    colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
    cmap = LinearSegmentedColormap.from_list("custom_blues", colors, N=256)
    
    # Plot the attention matrix with the custom colormap
    ax = sns.heatmap(
        attention,
        cmap=cmap,
        square=True,
        xticklabels=10,
        yticklabels=10,
        vmin=0,
        vmax=np.percentile(attention, 95),  # Cap at 95th percentile for better contrast
    )
    
    # Add a title with a professional look
    title = plt.title("Attention Visualization", fontsize=18, pad=20)
    title.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='#f0f0f0')])
    
    # Add labels
    plt.xlabel("Token Position", fontsize=14, labelpad=10)
    plt.ylabel("Token Position", fontsize=14, labelpad=10)
    
    # Improve the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label("Attention Weight", fontsize=14, labelpad=15)
    
    # Add grid lines for better readability
    ax.set_xticks(np.arange(0, len(attention), 10))
    ax.set_yticks(np.arange(0, len(attention), 10))
    

    
    # Improve overall appearance
    plt.tight_layout()
    
    # Save the figure with high resolution
    plt.savefig("benchmark/figures/attention_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Attention visualization saved as 'benchmark/figures/attention_visualization.png'")

    
    
if __name__ == "__main__":
    config = TransformerConfig.from_pretrained("AI4Protein/VenusPLM-300M", attn_impl="naive")
    tokenizer = VPLMTokenizer.from_pretrained("AI4Protein/VenusPLM-300M")
    model = TransformerForMaskedLM.from_pretrained("AI4Protein/VenusPLM-300M", config=config).eval().cuda()
    attention = show_attention("benchmark/data/swissprot/seqs.txt", model=model, tokenizer=tokenizer)
    plot_attention(attention)