import torch
import torch.nn.functional as F
import moeut
import profiles

batch_size = 8
context_window = 1024
vocab_size = 8000

tokens = torch.randint(0, vocab_size, (batch_size, context_window+1)).cuda()

model = moeut.MoEUTLM(vocab_size, **profiles.MoEUT_244M).cuda()
out = model(tokens[:, :-1])

print("Output shape: ", out.outputs.shape)
loss = F.cross_entropy(out.outputs.view(-1, vocab_size), tokens[:, 1:].flatten())
(loss + out.reg_loss).backward()

print("Loss: ", loss.item())