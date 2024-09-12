#!/bin/bash

pip install -r requirements.txt

g++ -c linear.cpp
g++ -o bin/linear.out linear.o

