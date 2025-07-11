#include <array>
#include <cstdint>
#include <string>

namespace nnue {
struct network {
   private:
    static constexpr int16_t INPUT_SIZE = 768;
    static constexpr int16_t INNER_SIZE = 2024;

    char checkValidInput(const std::string& fen);
    int16_t calculateIndex(const char& square, const char& pieceType, const char& color);
    std::array<char, INPUT_SIZE> calculateInput(const std::string& fen);
};
}  // namespace nnue