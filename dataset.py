import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len
        
        # Extra tokens
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id("<sos>")], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id("<eos>")], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id("<pad>")], dtype=torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx : int) -> Any:
        item = self.ds[idx]
        src_text = item['translation'][self.lang_src]
        tgt_text = item['translation'][self.lang_tgt]
        
        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Padding, counting:
        enc_num_padding_tokens = self.seq_len - len(src_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(tgt_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sequence length too short")

        # Add EOS and SOS tokens
        encoder_input = torch.cat(
            [self.sos_token,
            torch.tensor(src_tokens, dtype=torch.int64),
            self.eos_token,
            self.pad_token.repeat(enc_num_padding_tokens)
            ]
            )

        # Add SOS token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tgt_tokens, dtype=torch.int64),
                self.pad_token.repeat(dec_num_padding_tokens)
            ]
        )

        # Add SOS and EOS tokens
        label = torch.cat(
            [
                 torch.tensor(tgt_tokens, dtype=torch.int64),
                 self.eos_token,
                 self.pad_token.repeat(dec_num_padding_tokens)
            ]
        )

        assert encoder_input.size(0) == self.seq_len ,\
            f"encoder_input.size(0)={encoder_input.size(0)} != {self.seq_len}"

        assert decoder_input.size(0) == self.seq_len ,\
            f"decoder_input.size(0)={decoder_input.size(0)} != {self.seq_len}"

        assert label.size(0) == self.seq_len ,\
            f"label.size(0)={label.size(0)} != {self.seq_len}"

        return {
            'encoder_input': encoder_input, # (seq_len)
            'decoder_input': decoder_input, # (seq_len)
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, 1, seq_len) & (1, seq_len, seq_len)
            'label': label ,
            'src_text': src_text,
            'tgt_text': tgt_text
        }
        
def causal_mask(size):
    """
    Create a mask to prevent the decoder from peeking ahead
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
    
    