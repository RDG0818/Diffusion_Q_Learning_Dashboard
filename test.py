      
import torch
import torch.nn.functional as F

temp = 10**-6  # Example temperature
dummy_loss1 = torch.tensor(0.05)  # Example positive loss tensor
dummy_loss2 = torch.tensor(0.1)
x = -6
while True:
    x+=1
    temp *= 10    
    min_loss = torch.mean(-torch.logsumexp(-torch.stack([dummy_loss1, dummy_loss2]) * temp, dim=0) / temp)
    print(f"Temp: 10^{x} | Min:", min_loss.item()) # Should be positive
    if temp >= 10**5: break

    