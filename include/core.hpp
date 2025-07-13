#include <bitset>
#include <cstdint>
#include <string>

namespace nnue {
struct network {
   public:
    static constexpr int16_t INPUT_SIZE = 768;
    static constexpr int16_t INNER_SIZE = 2024;

    static char checkValidInput(const std::string& fen);
    static int16_t calculateIndex(const char& square, const char& pieceType, const char& color);
    static std::bitset<INPUT_SIZE + 1> calculateInput(const std::string& fen);
};
}  // namespace nnue