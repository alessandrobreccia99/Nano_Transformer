import torch
import torch.nn as nn
from torch.nn import functional as F

#----------------------------------------

# hyperparams
batch_size = 64
block_size = 64
n_emb = 54
max_iters = 4000
eval_interval = 500
learning_rate = 1e-3
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_head = 6
n_mul_att = 6
drop = 0.2

#----------------------------------------
torch.manual_seed(12)
# import text
with open('DivinaCommedia.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

# separate char and create alpha
chars = sorted(list(set(text)))
vocab_size = len(chars)

# convert to numbers (enco-deco)
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [ stoi[c] for c in s]
decode = lambda l:  ''.join([itos[i] for i in l ])

# create data train and test
data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# create arrays of context, from 1 char long to block size
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
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
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
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
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
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
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # resize idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx



if __name__ == "__main__":
        
        model = BigramLanguageModel()
        m = model.to(device)
        
        # create optimizer
        optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)
        
        
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
        
        m.eval()
        torch.save(m.state_dict(), 'BumbleBee_state_dict.pth')    
        
        context = torch.zeros((1,1), dtype=torch.long).to(device)
        print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
        