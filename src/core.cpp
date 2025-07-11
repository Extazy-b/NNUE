#include "../include/core.hpp"

char nnue::network::checkValidInput(const std::string& fen) {
    return 0;
}

// Pawn = 0, Knight = 1, Bishop, Rook, Queen, King

std::array<char, nnue::network::INPUT_SIZE> nnue::network::calculateInput(const std::string& fen) {
    char square = 0;  // 8a
    char pieceType = 0;
    char color = 0;
    char activeColor = 0;

    std::array<char, nnue::network::INPUT_SIZE> input;

    for (size_t i = 0; i < fen.length(); ++i) {
        if (fen[i] > 1 & fen[i] < 9) {
            square += fen[i];
            continue;
        }

        switch (tolower(fen[i])) {
            case '/':
                continue;

            case ' ':
                switch (fen[i + 1]) {
                    case 'w':
                        activeColor = 0;
                    case 'b':
                        activeColor = 1;
                }
                break;

            case 'p':
                pieceType = 0;
            case 'n':
                pieceType = 1;
            case 'b':
                pieceType = 2;
            case 'r':
                pieceType = 3;
            case 'q':
                pieceType = 4;
            case 'k':
                pieceType = 5;
        }
    }
};