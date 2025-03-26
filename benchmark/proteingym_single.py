from pathlib import Path
from Bio import SeqIO
from vplm.models.vplm.modeling_transformer import TransformerForMaskedLM
from vplm.models.vplm.tokenization_transformer import VPLMTokenizer
import torch
import pandas as pd
from scipy.stats import spearmanr

def read_seq_from_fasta(fasta_file):
    for record in SeqIO.parse(fasta_file, "fasta"):
        return str(record.seq)

def eval_transformer_proteingym():
    model = TransformerForMaskedLM.from_pretrained("AI4Protein/VenusPLM-300M")
    tokenizer = VPLMTokenizer.from_pretrained("AI4Protein/VenusPLM-300M")
    model.eval()
    model.cuda()
    fasta_dir = Path("benchmark/data/proteingym/fasta")
    mutant_dir = Path("benchmark/data/proteingym/mutant")
    mutant_files = list(Path(mutant_dir).rglob("*.csv"))
    rhos = []
    
    for mutant in mutant_files:
        mutant_file = str(mutant)
        fasta_file = str(fasta_dir / f"{mutant.stem}.fasta")
        inputs = tokenizer([read_seq_from_fasta(fasta_file)], return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()
        position_ids = torch.arange(input_ids.size(1)).unsqueeze(0).cuda()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=input_ids,
        )
        logits = outputs.logits
        logits = torch.log_softmax(logits, dim=-1).squeeze()[1:-1]
        vocab = tokenizer.get_vocab()
        df = pd.read_csv(mutant_file)
        scores = []
        for m in df["mutant"]:
            score = 0
            for sub in m.split(":"):
                wt, idx, mut = sub[0], int(sub[1:-1]) - 1, sub[-1]
                sub_score = logits[idx, vocab[mut]] - logits[idx, vocab[wt]]
                score += sub_score
            scores.append(score.item())
        rho = spearmanr(df["score"], scores)
        rhos.append(abs(rho.correlation))
        print(f"{mutant.stem}: {rho.correlation}")
    print(f"Average: {sum(rhos) / len(rhos)}")

if __name__ == "__main__":
    eval_transformer_proteingym()

