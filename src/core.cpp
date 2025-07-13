#include "../include/core.hpp"

#include <cstddef>
#include <cstdint>

char nnue::network::checkValidInput(const std::string& fen) {
    return 0;
}

int16_t nnue::network::calculateIndex(const char& square, const char& pieceType, const char& color) {
    return color * 64 * 6 + pieceType * 64 + square;
}

std::bitset<nnue::network::INPUT_SIZE + 1> nnue::network::calculateInput(const std::string& fen) {
    char square = 0;  // 8a square
    char pieceType = 0;
    char color = 0;
    std::bitset<INPUT_SIZE + 1> input;

    for (size_t i = 0; i < input.size(); i++) input[i] = 0;

    for (size_t i = 0; i < fen.length(); ++i) {
        if (fen[i] > '1' & fen[i] < '9') {
            square += fen[i] - 48;  // N empty squares
            continue;
        }

        switch (tolower(fen[i])) {
            case '/':
                continue;  // next line

            case ' ':  // end of position code
                switch (fen[i + 1]) {
                    case 'w':
                        input[INPUT_SIZE + 1] = 0;
                        break;
                    case 'b':
                        input[INPUT_SIZE + 1] = 1;
                        break;
                }
                break;

                if (fen[i] > 'a')
                    color = 1;  // lowercase
                else
                    color = 0;  // uppercase

                // Pawn = 0, Knight = 1, Bishop, Rook, Queen, King
            case 'p':
                pieceType = 0;
                break;
            case 'n':  // piece type codes
                pieceType = 1;
                break;
            case 'b':
                pieceType = 2;
                break;
            case 'r':
                pieceType = 3;
                break;
            case 'q':
                pieceType = 4;
                break;
            case 'k':
                pieceType = 5;
                break;
        }

        int16_t index = calculateIndex(square, pieceType, color);
        input[index] = 1;
    }

    return input;
};