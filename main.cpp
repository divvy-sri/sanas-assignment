
#include "linear.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

std::vector<std::vector<double>> readFile(std::string file_name) {
  std::vector<std::vector<double>> data;
  std::ifstream fin(file_name);
  std::string line;
  double num;
  if (fin) {
    while (std::getline(fin, line, '\n')) {
      std::vector<double> temp_vec;
      std::istringstream ss(line);
      while (ss >> num) {
        temp_vec.push_back(num);
      }
      data.emplace_back(temp_vec);
    }
  } else {
    std::cout << "File cannot be opened" << "\n";
  }
  fin.close();

  return data;
}

int main(int argc, char **argv) {
  int input_dim = 5;
  int output_dim = 1;
  int samples = 10;
  std::vector<std::vector<double>> weights = readFile(argv[1]);
  std::vector<std::vector<double>> bias = readFile(argv[2]);
  std::vector<std::vector<double>> input = readFile(argv[3]);
  Linear model = Linear(input_dim, output_dim, weights, bias);
  std::vector<std::vector<double>> normalized_input =
      model.normalizeInput(input);

  std::vector<std::vector<double>> output = model.forward(normalized_input);
  std::cout << "Cpp output:" << "\n";
  for (int i = 0; i < output.size(); i++) {
    for (int j = 0; j < output[0].size(); j++) {
      std::cout << output[i][j] << " ";
    }
    std::cout << "\n";
  }

  std::string output_filename = "data/cpp_output.txt";
  std::ofstream output_file(output_filename);
  if (output_file.is_open()) {
    for (auto &row : output) {
      for (auto &num : row) {
        output_file << num << " ";
      }
      output_file << "\n";
    }
    output_file.close();
  } else {
    std::cerr << "Error: Unable to open file for writing.\n";
  }
  return 0;
}