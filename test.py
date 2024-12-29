      
import torch
import torch.nn.functional as F

temp = 1  # Example temperature
dummy_loss1 = torch.rand(256, 1) * 10  # Example positive loss tensor
dummy_loss2 = torch.rand(256, 1) * 5 # Example positive loss tensor

min_loss = torch.mean(-torch.logsumexp(-torch.stack([dummy_loss1, dummy_loss2]) * temp, dim=0) / temp)
print(min_loss) # Should be positive
print(min_loss.min()) # Check if anything is negative.

    