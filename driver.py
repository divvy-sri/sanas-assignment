import random
import torch.random
import subprocess
from Linear import Linear
import numpy as np

# Parameters
seed = 12
input_dim = 50
output_dim = 10
samples = 10
torch.manual_seed(seed=seed)

# Generate random 
input = torch.rand(samples,input_dim,dtype=torch.double)
weights = torch.rand(output_dim,input_dim,dtype=torch.double)
bias = torch.rand(output_dim,dtype=torch.double)

# Save to files for c++
np.savetxt('input.txt', input.numpy())
np.savetxt('weights.txt', weights.numpy())
np.savetxt('bias.txt', bias.numpy())

# Forward pass with pytorch
model = Linear(input_dim,output_dim,weights=weights,bias=bias)
pytorch_output = model.Forward(input)
print("Pytorch output:",pytorch_output)

#Forward pass with c++
subprocess.run(["./linear.out", "weights.txt","bias.txt", "input.txt"]) 

#Load c++ output
cpp_output = torch.from_numpy(np.loadtxt("cpp_output.txt", dtype=np.double))


#Compare the outputs
if torch.allclose(pytorch_output, cpp_output, atol=1e-6):
    print("Outputs are close!")
else:
    print("Outputs differ!")