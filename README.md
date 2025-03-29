
# VenusPLM

![VenusPLM](VenusPLM.png){width=20%}
This repository introduces the VenusPLM, a protein language model that designed for protein representation and protein design.

## Pre-training

### Pre-training data

VenusPLM is pre-trained on VenusPod, a protein sequence dataset with over 9 billion sequences. 
The dataset is constructed by integrating multiple protein sequence databases, including UniRef100, OAS, NCBI, GOPC, MGnify, JGI and MEER, etc.

### Pre-training task

The pre-training task is to predict the missing residues or the substitution of the residues.

### Pre-training model

VenusPLM is a transformer model with RoPE and supports sequence packing and flash attention.
The sequence packing enables the model to encoding numerous protein sequences in efficient way. The flash attention can speed up the attention computation.

## Installation

VenusPLM requires Python 3.10 or higher. You can install VenusPLM and its dependencies using pip:

1. Install the required dependencies (If you have already installed the dependencies, you can skip this step):

   ```bash
   # Install PyTorch (>=2.5.0)
   pip3 install torch torchvision torchaudio
   
   # Install Transformers
   pip3 install transformers
   
   # Install flash-attn (Recommended but not required)
   pip install flash-attn --no-build-isolation
   ```

2. Install VenusPLM:

   ```bash
   pip install vplm
   ```

## VenusPLM For Protein Representation

### Model and tokenizer initialization

```python
from vplm import TransformerForMaskedLM, TransformerConfig
from vplm import VPLMTokenizer
import torch

config = TransformerConfig.from_pretrained("AI4Protein/VenusPLM-300M", attn_impl="sdpa") # or "flash_attn" if you have installed flash-attn
model = TransformerForMaskedLM.from_pretrained("AI4Protein/VenusPLM-300M", config=config)
tokenizer = VPLMTokenizer.from_pretrained("AI4Protein/VenusPLM-300M")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

Note that if you are in a region with strict firewalls, you may need to configure the huggingface mirror. For example, you can use [HF-Mirror](https://hf-mirror.com) by adding the following code in your python script:

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

or set the environment variable in your terminal:

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

### Encode a single protein sequence

```python
sequence = "MALWMRLLPLLALLALWGPDPAAA"
encoded_sequence = tokenizer(sequence, return_tensors="pt").to(device)

input_ids = encoded_sequence["input_ids"]
attention_mask = encoded_sequence["attention_mask"]

with torch.no_grad():
   outputs = model(
      input_ids=input_ids, 
      attention_mask=attention_mask,
      output_hidden_states=True
   )

hidden_states = outputs.hidden_states[-1].squeeze()
print(hidden_states)
```

The output will be a tensor containing the hidden states for each token in the sequence:

```python
tensor([[-0.0067,  0.0164, -0.0360,  ..., -0.0135, -0.0005, -0.0574],
        [ 0.0641, -0.0427, -0.1948,  ...,  0.0465,  0.0883, -0.1913],
        [-0.0226,  0.0628, -0.1420,  ...,  0.0705,  0.1389, -0.4280],
        ...,
        [ 0.0142,  0.1574, -0.0865,  ..., -0.0621,  0.1190, -0.1893],
        [-0.0053,  0.1854, -0.0482,  ..., -0.0396,  0.1402, -0.2690],
        [-0.0125, -0.1690, -0.0629,  ...,  0.0205, -0.0212, -0.1900]])
```

### Encode a batch of protein sequences

```python
sequences = ["MALWMRLLPAA", "MALWMRLLPLLALLALWGPDPAAA", "MALWMRLLPLLWGPDPAAA"]
encoded_sequences = tokenizer(sequences, return_tensors="pt", padding=True).to(device)

input_ids = encoded_sequences["input_ids"]
attention_mask = encoded_sequences["attention_mask"]

with torch.no_grad():
   outputs = model(
      input_ids=input_ids, 
      attention_mask=attention_mask,
      output_hidden_states=True
   )

hidden_states = outputs.hidden_states[-1]
print(hidden_states)
```

### Encode protein sequences with sequence packing (if the flash-attn is not installed)

```python
sequence_group = ["METLT", "MYTYDSF", "MGCTLNWVGIQD"]

encoded_sequences = tokenizer(
   sequence_group,  
   padding=False,
   return_attention_mask=False,
   return_length=True,
)

# concat the input_ids along the length dimension
input_ids = torch.cat(
   [torch.tensor(seq, device=device) for seq in encoded_sequences["input_ids"]],
   dim=0
).to(torch.long).unsqueeze(0) # [1, L]

# concat the attention_mask along the length dimension and add sequence offset
attention_mask = torch.cat(
   [torch.ones(l, device=device) + idx for idx, l in 
   enumerate(encoded_sequences["length"])],
   dim=0
).to(torch.long).unsqueeze(0) # [1, L]

print(input_ids)
# tensor([[ 1, 19,  8, 10,  3, 10,  2,  1, 19, 18, 10, 18, 12,  7, 17,  2,  1, 19,
#        5, 22, 10,  3, 16, 21,  6,  5, 11, 15, 12,  2]], device='cuda:0')

print(attention_mask)
# tensor([[ 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
#         3, 3, 3, 3, 3, 3]], device='cuda:0')

with torch.no_grad():
   outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      output_hidden_states=True,
      output_attentions=False, # sdpa does not support output_attentions
   )

hidden_states = outputs.hidden_states[-1]
print(hidden_states)
```

### Encode protein sequences with sequence packing (if the flash-attn is installed)

```python
from vplm import TransformerForMaskedLM, TransformerConfig
from vplm import VPLMTokenizer
import torch

config = TransformerConfig.from_pretrained("AI4Protein/VenusPLM-300M", attn_impl="flash_attn") 
model = TransformerForMaskedLM.from_pretrained("AI4Protein/VenusPLM-300M", config=config)
tokenizer = VPLMTokenizer.from_pretrained("AI4Protein/VenusPLM-300M")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.to(torch.bfloat16) # Flash attention only supports bfloat16

sequence_group = ["METLT", "MYTYDSF", "MGCTLNWVGIQD"]

encoded_sequences = tokenizer(
   sequence_group,  
   padding=False,
   return_attention_mask=False,
   return_length=True,
)

# concat the input_ids along the length dimension
input_ids = torch.cat(
   [torch.tensor(seq, device=device) for seq in encoded_sequences["input_ids"]],
   dim=0
).to(torch.long).unsqueeze(0) # [1, L]

lengths = [encoded_sequences["length"], ]

print(input_ids)
# tensor([[ 1, 19,  8, 10,  3, 10,  2,  1, 19, 18, 10, 18, 12,  7, 17,  2,  1, 19,
#        5, 22, 10,  3, 16, 21,  6,  5, 11, 15, 12,  2]], device='cuda:0')

print(lengths)
# [[7, 9, 14]]

with torch.no_grad():
   with torch.autocast(device.type, dtype=torch.bfloat16):
      outputs = model(
         input_ids=input_ids,
         lengths=lengths,
         output_hidden_states=True,
         output_attentions=False,
      )

hidden_states = outputs.hidden_states[-1]
print(hidden_states)
```
See [benchmark/infer_speed.py](benchmark/infer_speed.py) for more comprehensive details and inference throughput comparison.

### Watch the sequence packing attention

```python
from vplm import TransformerForMaskedLM, TransformerConfig
from vplm import VPLMTokenizer
import torch

config = TransformerConfig.from_pretrained("AI4Protein/VenusPLM-300M", attn_impl="naive") 
model = TransformerForMaskedLM.from_pretrained("AI4Protein/VenusPLM-300M", config=config)
tokenizer = VPLMTokenizer.from_pretrained("AI4Protein/VenusPLM-300M")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

sequence_group = ["METLT", "MYTYDSF", "MGCTLNWVGIQD"]

encoded_sequences = tokenizer(
   sequence_group,  
   padding=False,
   return_attention_mask=False,
   return_length=True,
)

# concat the input_ids along the length dimension
input_ids = torch.cat(
   [torch.tensor(seq, device=device) for seq in encoded_sequences["input_ids"]],
   dim=0
).to(torch.long).unsqueeze(0) # [1, L]

# concat the attention_mask along the length dimension and add sequence offset
attention_mask = torch.cat(
   [torch.ones(l, device=device) + idx for idx, l in 
   enumerate(encoded_sequences["length"])],
   dim=0
).to(torch.long).unsqueeze(0) # [1, L]

with torch.no_grad():
   outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      output_hidden_states=True,
      output_attentions=True,
   )

attention_weights = outputs.attentions[-1].squeeze().mean(dim=0)
print(attention_weights)

# If you want to visualize the attention weights, you can use the following code:
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Convert attention weights to numpy for visualization
attention_np = attention_weights.cpu().numpy()
colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
cmap = LinearSegmentedColormap.from_list("custom_blues", colors, N=256)
# Create a figure
plt.figure(figsize=(10, 8))

# Create a heatmap using seaborn
sns.heatmap(
    attention_np,
    cmap=cmap,
    xticklabels=5,
    yticklabels=5,
    cbar_kws={"label": "Attention Weight"}
)

plt.tight_layout()
plt.savefig("attention_visualization.png", dpi=300)
plt.close()
```

![VenusPLM](benchmark/figures/readme_attn.png)

See [benchmark/sequence_pack_attention_vis.py](benchmark/sequence_pack_attention_vis.py) for more comprehensive details.

## VenusPLM Language Model For Protein Design (Mutant Effect Prediction)

```python
from vplm import TransformerForMaskedLM, TransformerConfig
from vplm import VPLMTokenizer
import torch
import pandas as pd

wild_type_name = "ProteinX"
wild_type = "MRFAVVLLAVVLTFANAARAGETLTVYTYDSFVADWGPGPAIKESFEGECGCTLNWVGIQDGVAILNRLKLEGSSTKADVILGLDTNLLTEAKATGLLAPHGADLSGVKLP"
amino_acids = "LAGVSERTIDPKQNFYMHWC"

tokenizer = VPLMTokenizer.from_pretrained("AI4Protein/VenusPLM-300M")
model = TransformerForMaskedLM.from_pretrained("AI4Protein/VenusPLM-300M")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

encoded_wild_type = tokenizer(wild_type, return_tensors="pt").to(device)
with torch.no_grad():
   outputs = model(
      input_ids=encoded_wild_type["input_ids"],
      attention_mask=encoded_wild_type["attention_mask"],
      output_hidden_states=False,
   )

logits = outputs.logits.log_softmax(dim=-1).squeeze()[1:-1]

mutants = []
mutant_seqs = []
scores = []
for idx, wt_aa in enumerate(wild_type):
   for mt_aa in amino_acids:
      if mt_aa == wt_aa:
         continue
      mutant_seq = wild_type[:idx] + mt_aa + wild_type[idx+1:]
      mutant = f"{wt_aa}{idx+1}{mt_aa}"
      mt_token = tokenizer.get_vocab()[mt_aa]
      wt_token = tokenizer.get_vocab()[wt_aa]
      score = logits[idx, mt_token] - logits[idx, wt_token]
      mutants.append(mutant)
      mutant_seqs.append(mutant_seq)
      scores.append(score.item())

df = pd.DataFrame({
   "mutant": mutants,
   "mutant_seq": mutant_seqs,
   "score": scores,
})
df = df.sort_values(by="score", ascending=False)
print(df)
df.to_csv(f"mutant_prediction_{wild_type_name}.csv", index=False)
```

```python
     mutant                                         mutant_seq      score
903    G48K  MRFAVVLLAVVLTFANAARAGETLTVYTYDSFVADWGPGPAIKESF...   1.570629
287    N16G  MRFAVVLLAVVLTFAGAARAGETLTVYTYDSFVADWGPGPAIKESF...   0.955346
837    S45A  MRFAVVLLAVVLTFANAARAGETLTVYTYDSFVADWGPGPAIKEAF...   0.677591
894    G48A  MRFAVVLLAVVLTFANAARAGETLTVYTYDSFVADWGPGPAIKESF...   0.539928
898    G48R  MRFAVVLLAVVLTFANAARAGETLTVYTYDSFVADWGPGPAIKESF...   0.319188
...     ...                                                ...        ...
453    L24H  MRFAVVLLAVVLTFANAARAGETHTVYTYDSFVADWGPGPAIKESF... -14.204397
1511   V80K  MRFAVVLLAVVLTFANAARAGETLTVYTYDSFVADWGPGPAIKESF... -14.446470
484    V26P  MRFAVVLLAVVLTFANAARAGETLTPYTYDSFVADWGPGPAIKESF... -14.798222
445    L24D  MRFAVVLLAVVLTFANAARAGETDTVYTYDSFVADWGPGPAIKESF... -14.828167
449    L24N  MRFAVVLLAVVLTFANAARAGETNTVYTYDSFVADWGPGPAIKESF... -15.079876
```

See [benchmark/proteingym_single.py](benchmark/proteingym_single.py) for more comprehensive single protein design performance.

## Acknowledgements

This project is supported by [Liang's Group](https://ins.sjtu.edu.cn/people/lhong/index.html) of Shanghai Jiao Tong University.

## Citation

Mingchen Li, Bozitao Zhong, Pan Tan, Liang Hong. VenusPLM: A Protein Language Model for Protein Representation and Design [EB/OL] (2025) <https://github.com/AI4Protein/VenusPLM>.
