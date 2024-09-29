#include "linear.h"
#include <cmath>
#include <math.h>

Linear::Linear(int input_dim, int output_dim,
               std::vector<std::vector<double>> weights,
               std::vector<std::vector<double>> bias)
    : input_dim(input_dim), output_dim(output_dim), weights(weights),
      bias(bias) {}

Linear::~Linear() = default;

std::vector<std::vector<double>>
Linear::normalizeInput(std::vector<std::vector<double>> &input) {
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
Linear::forward(std::vector<std::vector<double>> &input) {
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
