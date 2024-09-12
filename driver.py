import random
import torch.random
import subprocess
from Linear import Linear
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Driver code to run forward pass with pytorch and c++')
parser.add_argument('input_dim', help='Input dimensions',type=int)
parser.add_argument('output_dim', help='Ouput dimensions',type=int)
parser.add_argument('seed', help='Seed for random number generation',type=int)
args = parser.parse_args()

# parser.add

# Parameters
input_dim = args.input_dim
output_dim = args.output_dim
seed = args.seed
samples = 10
torch.manual_seed(seed=seed)

# Generate random 
input = torch.rand(samples,input_dim,dtype=torch.double)
weights = torch.rand(output_dim,input_dim,dtype=torch.double)
bias = torch.rand(output_dim,dtype=torch.double)

# Save to files for c++
np.savetxt('data/input.txt', input.numpy())
np.savetxt('data/weights.txt', weights.numpy())
np.savetxt('data/bias.txt', bias.numpy())

# Forward pass with pytorch
model = Linear(input_dim,output_dim,weights=weights,bias=bias)
pytorch_output = model.Forward(input)
print("Pytorch output:\n",pytorch_output)

#Forward pass with c++
subprocess.run(["./bin/linear.out", "data/weights.txt","data/bias.txt", "data/input.txt"]) 

#Load c++ output
cpp_output = torch.from_numpy(np.loadtxt("data/cpp_output.txt", dtype=np.double))


#Compare the outputs
if torch.allclose(pytorch_output, cpp_output, atol=1e-6):
    print("\nOutputs are close!\n")
else:
    print("\nOutputs differ!\n")