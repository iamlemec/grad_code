# text related tools

import re
import torch
import torch.nn.functional as F

# strip punctuation and merge whitespace
def process_text(s):
    # merge punctuation
    s = re.sub(r'(;|\-\-)', r',', s)
    s = re.sub(r'([\-\[\]])', ' ', s)
    s = re.sub(r'([,\.!\?"\(\)])', r' \1 ', s)
    s = re.sub(r'\r', ' ', s)

    # merge whitespace
    s = re.sub(r'(?<!\n)\n(?!\n)', ' ', s)
    s = re.sub(r'\n{2,}', '\n', s)
    s = re.sub(r' {2,}', ' ', s)

    # strip lines
    s = re.sub(r'(^ +| +$)', r'', s, flags=re.MULTILINE)

    return s.lower()

# generate overlapping sequences with l-length history and single outcome
def generate_sequences(vec, l):
    n, t = vec.shape
    k = t - l
    index0 = torch.arange(l).unsqueeze(0) + torch.arange(k).unsqueeze(1)
    index = index0.flatten().expand(n, k*l)
    series = torch.gather(vec, 1, index).reshape(n*k, l)
    target = vec[:, l:].reshape(n*k)
    return series, target

# get total module parameter count
def total_params(mod):
    return sum([p.numel() for p in mod.parameters()])

# get batch indices
def batch_indices(length, batch_size):
    return [(i, min(i+batch_size, length)) for i in range(0, length, batch_size)]

# l2 normalize
def l2_normalize(x, dim=1):
    return x / x.norm(dim=dim, keepdim=True)

# ensure a list is of least length n by padding it
def pad_list(x, n, v=0):
    return x[-n:] + [v]*max(0, n-len(x))

# sample index from a tensor of logits
def sample_logits(logits, temp=1.0):
    probs = F.softmax(logits/temp)
    index = probs.multinomial(num_samples=1)
    return index.item()
