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
         std::vector<std::vector<double>> bias);
  ~Linear();

  std::vector<std::vector<double>>
  normalizeInput(std::vector<std::vector<double>> &input);


  std::vector<std::vector<double>>
  forward(std::vector<std::vector<double>> &input); 


};