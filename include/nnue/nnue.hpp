#include <array>
#include <cstdint>
#include <string>

namespace nnue {
const int16_t INPUT_SIZE = 768;
const int16_t INNER_SIZE = 2024;

class NNUE {
public:
  NNUE(std::string model_path);
  int16_t forward(const std::string &input) const;
  void load_weights(const std::string &file);

private:
  int calculateIndex(int square, int piceType, int side);
  std::array<int16_t, INNER_SIZE> inputLayer(std::array<char, INPUT_SIZE>);
  int
}
} // namespace nnue
