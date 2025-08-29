
from typing import List, Tuple, Dict, Any
import itertools
import torch
from transformers import AutoModel, AutoTokenizer

class AwesomeAligner:
    """
    Word- and subword-level alignment using aneuraz/awesome-align-with-co.
    Returns both subword alignment pairs and deduplicated word alignment pairs (src_word_idx, tgt_word_idx).
    """
    def __init__(self, model_name: str = "aneuraz/awesome-align-with-co", device: str | None = None,
                 align_layer: int = 8, threshold: float = 1e-3):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.align_layer = align_layer
        self.threshold = threshold

    def _prep(self, src: str, tgt: str):
        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
        token_src = [self.tokenizer.tokenize(w) for w in sent_src]
        token_tgt = [self.tokenizer.tokenize(w) for w in sent_tgt]
        wid_src = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src]
        wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
        ids_src = self.tokenizer.prepare_for_model(
            list(itertools.chain(*wid_src)), return_tensors='pt',
            model_max_length=self.tokenizer.model_max_length, truncation=True
        )['input_ids']
        ids_tgt = self.tokenizer.prepare_for_model(
            list(itertools.chain(*wid_tgt)), return_tensors='pt',
            model_max_length=self.tokenizer.model_max_length, truncation=True
        )['input_ids']
        sub2word_src = []
        for i, word_list in enumerate(token_src):
            sub2word_src += [i for _ in word_list]
        sub2word_tgt = []
        for i, word_list in enumerate(token_tgt):
            sub2word_tgt += [i for _ in word_list]
        return (sent_src, sent_tgt, ids_src, ids_tgt, sub2word_src, sub2word_tgt)

    def align(self, src: str, tgt: str) -> Dict[str, Any]:
        sent_src, sent_tgt, ids_src, ids_tgt, s2w_src, s2w_tgt = self._prep(src, tgt)
        with torch.no_grad():
            out_src = self.model(ids_src.unsqueeze(0).to(self.device), output_hidden_states=True)[2][self.align_layer][0, 1:-1]
            out_tgt = self.model(ids_tgt.unsqueeze(0).to(self.device), output_hidden_states=True)[2][self.align_layer][0, 1:-1]
            dot = torch.matmul(out_src, out_tgt.transpose(-1, -2))
            soft_srctgt = torch.nn.Softmax(dim=-1)(dot)
            soft_tgtsrc = torch.nn.Softmax(dim=-2)(dot)
            inter = (soft_srctgt > self.threshold) * (soft_tgtsrc > self.threshold)
        sub_pairs = torch.nonzero(inter, as_tuple=False).tolist()
        word_pairs = {(s2w_src[i], s2w_tgt[j]) for i, j in sub_pairs}
        return {
            "src_words": sent_src,
            "tgt_words": sent_tgt,
            "subword_align": sub_pairs,             # list[[sub_i, sub_j], ...]
            "word_align": sorted(list(word_pairs))  # list[(src_word_i, tgt_word_j), ...]
        }
