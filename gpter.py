#!/home/user/mambaforge/envs/env/bin/python

import tiktoken, time
from tinygrad import Tensor, nn, dtypes, TinyJit, Context, GlobalCounters, Device
from tinygrad.helpers import getenv
import numpy as np

# Hyperparameters
Tensor.manual_seed(1337)
batch_size = 64 # how many independent sequences we will process in parallel 
block_size = 256 # what is the maximum context length for predictions?
eval_iters = 200
eval_interval = 500
learning_rate = 3e-4
max_iters = 10000
n_embed = 384

class Head:
    """ one head of self-attention """

    def __init__(self, head_size):
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.tril = Tensor.tril(Tensor.ones(block_size,block_size))
        self.tril.requires_grad = False

    def __call__(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        tril = Tensor.tril(Tensor.ones(T,T))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = wei.softmax()

        v = self.value(x)
        out  = wei @ v
        return out

class MultiHeadAttention:
    def __init__(self, num_heads, head_size):
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = nn.Linear(n_embed, n_embed)

    def __call__(self, x):
        first = self.heads[0](x)
        out = first.cat(*[h(x) for h in self.heads[1:]], dim=-1)
        out = self.proj(out)
        return out

class FeedForward():
    """ simpel linear layer followed by a non-linearity """

    def __init__(self, n_embed):
        self.w1 = nn.Linear(n_embed, 4 * n_embed)
        self.w2 = nn.Linear(4 * n_embed, n_embed) # projection layer

    def __call__(self, x): 
        return self.w2(self.w1(x).relu())

class Block():
    """ transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head):
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def __call__(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel:
    def __init__(self):
        self.token_embedding_table = nn.Embedding(vocab_size=vocab_size, embed_size=n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_heads = MultiHeadAttention(4, n_embed//4)
        self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(Tensor.arange(T)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.sa_heads(x) # one head of self attention
        x = self.ffwd(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            
            loss = logits.cross_entropy(targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = logits.softmax(axis=1)
            idx_next = probs.multinomial(num_samples=1)
            idx = idx.cat(idx_next, dim=1)
        return idx

def get_batch(split):
    data = train_data if split == 'train' else val_data
    #rand = Tensor.randint(high=(len(data) - block_size), requires_grad=False).item()
    rand = np.random.randint(0,(len(data)-block_size*batch_size))
    #rand = 1337

    x = data[rand:rand+block_size*batch_size].view(batch_size, block_size)
    y = data[rand+1:rand+block_size*batch_size+1].view(batch_size, block_size)

    return x, y

@TinyJit
@Tensor.test()
def estimate_loss():
    Tensor.no_grad, Tensor.training = False, True
    out = {}
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses.append(loss.item())
        out[split] = Tensor(losses).mean()
    Tensor.no_grad, Tensor.training = True, False
    return out
    
@TinyJit
@Tensor.test()
def test(xb, yb):
    logits, loss = m(xb, yb)
    return loss.realize()


@TinyJit
@Tensor.train()
def step(xb, yb):
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    #optimizer.step()
    return loss.realize(*optimizer.schedule_step())

# show Devices
GPUS = tuple(f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 1)))
#print(Device.DEFAULT)

# import data
text = open('wiki_all.txt', 'r').read()

# generate embedings
#encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.get_encoding("r50k_base")
text_encoded = encoding.encode(text)
vocab_size = encoding.n_vocab

data = Tensor(text_encoded, dtype=dtypes.long, requires_grad=True)
# split into training and validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

m = BigramLanguageModel()
optimizer = nn.optim.AdamW(nn.state.get_parameters(m), lr=learning_rate)
losses, times, speed = [], [], []
print(sum(p.numel() for p in nn.state.get_parameters(m))/1e6, 'M parameters')

# Traning phase
for steps in range(max_iters):
    #print(steps) 
    # sample data
    xb, yb = get_batch('train')

    # evaluate loss
    GlobalCounters.reset()
    t0 = time.time()
    loss = step(xb.contiguous(), yb.contiguous())
    Device[Device.DEFAULT].synchronize()
    t1 = time.time()

    #losses.append(loss.item())
    times.append(t1-t0)
    speed.append(batch_size*block_size/(t1-t0))

    # track stats
    if steps % eval_interval == 0:
        Tensor.no_grad, Tensor.training = True, False

        for k in range(eval_iters):
            # sample data
            xb, yb = get_batch('train')

            # evaluate loss
            loss = test(xb.contiguous(), yb.contiguous())

            # save stats
            losses.append(loss.item())
        loss_mean = Tensor(losses).mean().item()
        time_mean = Tensor(times).mean().item()
        speed_mean = Tensor(speed).mean().item()
        print(f"step {steps}, mean loss: {loss_mean:.4f}, training time: {time_mean:.4f}s, training speed: {speed_mean:.4f} tok/s")
        losses, times, speed = [], [], []
        Tensor.no_grad, Tensor.training = False, True

    #print('++++++++++++++++++++', steps, ':', loss.item())

# eval
Tensor.no_grad, Tensor.training = True, False
losses = []
for k in range(eval_iters):
    # sample data
    xb, yb = get_batch('val')

    # evaluate loss
    loss = test(xb.contiguous(), yb.contiguous())

    # save stats
    losses.append(loss.item())

loss_mean = Tensor(losses).mean().item()
print(f"EVAL LOSS: {loss_mean:.4f}")

print("SHOW ME WHAT YOU GOT:")
print(encoding.decode(m.generate(Tensor.zeros((1,1), dtype=dtypes.long), max_new_tokens=100)[0].tolist()))
