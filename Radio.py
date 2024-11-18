import torch
from BumbleBee import decode
from BumbleBee import BigramLanguageModel

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

model = BigramLanguageModel()
model.load_state_dict(torch.load('BumbleBee_state_dict.pth'))
model.eval()
model.to(device)

context = torch.zeros((1,1), dtype=torch.long).to(device)

torch.manual_seed(123)
with open('DiVinoCommedia.txt', 'w', encoding = 'utf-8') as f:
        
    f.write(decode(model.generate(context, max_new_tokens=1000, T=0.7)[0].tolist()))
