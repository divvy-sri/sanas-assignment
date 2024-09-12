# Sanas assignment

## 

This assisgnment has the following objective:

Make a PyTorch module which receives an input, the Module should do these things

1) Normalize the input along each column [Sum of squares of all elements in a column = 1]
2) Implement Linear Layer

Generate 10 random input samples and pass it through this Module, save the results.

Now, do the same implementation using C++ and assert the results from Pytorch and C++ to be close [or same]

## 

My implementation is tested on ubuntu 22.04.4 LTS with python version 3.10.12, and g++ compiler version 11.4.0.

I would create a virtual environment to run this assignment. Install virtualenv.

Clone this repository, enter the folder, and run the following:

```bash
virtualenv assessment

source assessment/bin/activate
```

To install python libraries and compile the c++ file run:

```bash
bash setup.sh

```

Driver.py is the driver code that generates the inputs, weights, and biases, does forward pass with pytorch and c++, and compares the outputs of both.
Linear.py has the pytorch module and Linear.cpp has the cpp module and the main function.

```python
python driver.py input_dimension output_dimension seed
```

example:

```python
python driver.py 35 25 50
```

It should print whether the outputs are close or not.
