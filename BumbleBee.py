import torch
import torch.nn as nn
import re
from torch.nn import functional as F

#----------------------------------------
def get_stats(tokens):
    counts = {}

    for pair in zip(tokens,tokens[1:]):
        if pair[0] == tok_encode('<sos>')[0] or pair[0] == tok_encode('<eos>')[0] or pair[1] == tok_encode('<sos>')[0] or pair[1] == tok_encode('<eos>')[0]:
            counts[pair] = 0
        counts[pair] = counts.get(pair, 0) + 1
    return counts \

def merge(ids, pair, idx):
    newids = []
    i=0
    while i < len(ids):
        # if we are not at the very last position AND the pair matches, replace it
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
          newids.append(idx)
          i += 2
        else:
          newids.append(ids[i])
          i += 1 
    return newids

#----------------------------------------
# hyperparams
batch_size = 64
context_len = 32 
vocab_size = 64
n_emb = 128 
max_iters = 3500
eval_interval = 250  #every how many iters the loss is printed
learning_rate = 8e-3
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200 
n_head = 4 
n_mul_att = 8 
drop = 0.1 # dropout percentage

#----------------------------------------
torch.manual_seed(12)
# import text
with open('dialogs_clean.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

# separate char and create alpha
chars = sorted(list(set(text)))
chars.append('<sos>')
chars.append('<eos>')
#vocab_size = len(chars)

# convert to numbers (enco-deco)
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
tok_decode = lambda i: [itos[c] for c in i]
def tok_encode(s):
    s = re.sub(r'<sos>', '+', s)
    s = re.sub(r'<eos>', '_', s)

    d = {'+':stoi['<sos>'],'_':stoi['<eos>']}
    return [d[c] if c in ['+','_'] else stoi[c] for c in s]

# TOKENS TRAINING TIME
#tokens = list(text.encode("utf-8"))
tokens = tok_encode(text)
num_merges = vocab_size - len(chars)
ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = len(chars) + i
  ids = merge(ids, pair, idx)
  merges[pair] = idx

# TOKENS INFERENCE TIME 
def encode(text, tokens):
    # given a string, return list of integers (the tokens)
    while len(tokens) >= 2:
      stats = get_stats(tokens)
      pair = min(stats, key=lambda p: merges.get(p, float("inf")))
      if pair not in merges:
        break # nothing else can be merged
      idx = merges[pair]
      tokens = merge(tokens, pair, idx)
    return tokens    

# DECODING 
vocab = {idx: itos[idx] for idx in range(len(chars))}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
  # given ids (list of integers), return Python string
  #tokens = b"".join(vocab[idx] for idx in ids)
  #text = tokens.decode("utf-8", errors="replace")
  text = "".join(vocab[idx] for idx in ids)
  return text

# create data train and test
data = torch.tensor(encode(text, tokens), dtype=torch.long, device=device)
n = int(0.95*len(data))
train_data = data[:n]
val_data = data[n:]

# create arrays of context, from 1 char long to block size
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_len, (batch_size,))
    x = torch.stack([data[i:i+context_len] for i in ix])
    y = torch.stack([data[i+1:i+context_len+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {} 
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out    

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_len,context_len)))
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        B,T,C = x.shape

        k = self.key(x) # (B,T, H)
        q = self.query(x) # (B, T, H)

        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,H) @ (B, H, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.drop(wei)

        v = self.value(x) # (B, T, H)
        out = wei @ v # (B,T,T) @ (B, T, H) -> (B, T, H)
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList( Head(head_size) for _ in range(num_heads) )
        self.proj = nn.Linear(n_emb, n_emb)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        out = self.drop(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential( nn.Linear(n_emb, n_emb * n_head), nn.ReLU(), nn.Linear(n_head * n_emb, n_emb), nn.Dropout(p=drop))

    def forward(self, x):
        return self.net(x)    

class Block(nn.Module):

    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) 
        return x 

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb) 
        self.position_embedding_table = nn.Embedding(context_len, n_emb)
        self.blocks = nn.Sequential(*[Block(n_emb, n_head=n_head) for _ in range(n_mul_att)],)
        self.l_n = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size) 
        self.drop = nn.Dropout(drop)         

    def forward(self, idx, targets = None):
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensor
        tok_emb = self.token_embedding_table(idx)   # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = self.drop(tok_emb) + self.drop(pos_emb) # (B, T, C)
        x = self.blocks(x)
        x = self.l_n(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:    
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens, T):
        # idx is (B, T) array of indices in the current context
        len_input = idx.shape[1]
        for _ in range(max_new_tokens):
            # resize idx to the last context_len tokens
            idx_cond = idx[:, -context_len:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Apply temperature transform
            probs = probs**(1/T)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            if idx_next[0][0] == tok_encode('<eos>')[0]:
                break

        return idx[:, len_input:][0]
    

 

if __name__ == "__main__":
        
        model = BigramLanguageModel()
        m = model.to(device)
        print("The model has", sum(p.nelement() for p in m.parameters()) , "parameters" )
        
        # create optimizer
        optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        
        
        for iter in range(max_iters):
            
            if iter % eval_interval == 0:
                losses = estimate_loss()
                print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
        
        
            # sample a batch of data
            xb, yb = get_batch('train')
        
            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        m.eval()
        torch.save(m.state_dict(), 'BumbleBee_state_dict.pth')    
        
        #context = torch.zeros((1,1), dtype=torch.long).to(device)
        #print(decode(m.generate(context, max_new_tokens=1000, T=1.0)[0].tolist()))
        