#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

class Linear {

private:
  int input_dim;
  int output_dim;
  std::vector<std::vector<double>> weights;
  std::vector<std::vector<double>> bias;

public:
  Linear(int input_dim, int output_dim,
         std::vector<std::vector<double>> weights,
         std::vector<std::vector<double>> bias)
      : input_dim(input_dim), output_dim(output_dim), weights(weights),
        bias(bias) {}

  std::vector<std::vector<double>>
  normalizeInput(std::vector<std::vector<double>> &input) {
    int row_size = input.size();
    int col_size = input[0].size();
    std::vector<std::vector<double>> normalized_input = input;
    for (int col = 0; col < col_size; col++) {
      double sum_of_squares = 0;
      for (int row = 0; row < row_size; row++) {
        sum_of_squares += input[row][col] * input[row][col];
      }
      double norm_factor = std::sqrt(sum_of_squares);
      double check_sum = 0;
      if (norm_factor > 0) {
        for (int row = 0; row < row_size; row++) {
          normalized_input[row][col] /= norm_factor;
        }
      }
    }
    return normalized_input;
  }

  std::vector<std::vector<double>>
  forward(std::vector<std::vector<double>> &input) {
    int w_rows = this->weights.size();
    int w_cols = this->weights[0].size();
    std::vector<std::vector<double>> weight_transpose(
        w_cols, std::vector<double>(w_rows, 0.0));
    std::vector<std::vector<double>> output(
        input.size(), std::vector<double>(weight_transpose[0].size(), 0.0));
    int output_rows = output.size();
    int output_cols = output[0].size();

    for (int i = 0; i < w_rows; ++i) {
      for (int j = 0; j < w_cols; ++j) {
        weight_transpose[j][i] = weights[i][j];
      }
    }

    for (int i = 0; i < output_rows; i++) {
      for (int j = 0; j < output_cols; j++) {
        for (int k = 0; k < input[0].size(); k++) {
          output[i][j] += input[i][k] * weight_transpose[k][j];
        }
        output[i][j] += bias[j][0];
      }
    }

    return output;
  }
};

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
    for (auto& row : output) {
      for (auto& num : row) {
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