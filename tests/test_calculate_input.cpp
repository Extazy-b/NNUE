#include <gtest/gtest.h>

#include <bitset>
#include <iostream>
#include <string>
#include <vector>

#include "../include/core.hpp"

using namespace nnue;

TEST(CalculateInputTest, StartPosition) {
    std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1";
    auto result = network::calculateInput(fen);

    EXPECT_EQ(result[network::INPUT_SIZE], 0);  // белые ходят

    int16_t whitePawnIndex = network::calculateIndex(9, 0, 0);   // e2
    int16_t blackRookIndex = network::calculateIndex(63, 3, 1);  // a8

    EXPECT_EQ(result[whitePawnIndex], 1);
    EXPECT_EQ(result[blackRookIndex], 1);
}

TEST(CalculateInputTest, EmptyBoardBlackMove) {
    std::string fen = "8/8/8/8/8/8/8/8 b - - 0 1";
    auto result = network::calculateInput(fen);

    for (int i = 0; i < network::INPUT_SIZE; ++i) EXPECT_EQ(result[i], 0);

    EXPECT_EQ(result[network::INPUT_SIZE], 1);  // чёрные ходят
}

TEST(CalculateInputTest, SingleWhiteQueen) {
    std::string fen = "8/8/8/8/8/8/3Q4/8 w - - 0 1";
    auto result = network::calculateInput(fen);

    int square = 11;    // d2
    int pieceType = 4;  // Queen
    int color = 0;      // White

    int16_t index = network::calculateIndex(square, pieceType, color);
    EXPECT_EQ(result[index], 1);

    for (int i = 0; i < network::INPUT_SIZE; ++i) {
        if (i != index) EXPECT_EQ(result[i], 0);
    }

    EXPECT_EQ(result[network::INPUT_SIZE], 0);
}

TEST(CalculateInputTest, AllPieceTypes) {
    std::string fen = "k7/1r6/2b5/3q4/4n3/5p2/6P1/7K w - - 0 1";
    auto result = network::calculateInput(fen);

    struct PieceInfo {
        char square;
        char pieceType;
        char color;
    };

    std::vector<PieceInfo> expected = {
        {0, 5, 1},   // k на a8
        {9, 3, 1},   // r на b7
        {18, 2, 1},  // b на c6
        {27, 4, 1},  // q на d5
        {36, 1, 1},  // n на e4
        {45, 0, 1},  // p на f3
        {54, 0, 0},  // P на g2
        {63, 5, 0},  // K на h1
    };

    for (const auto& p : expected) {
        int16_t index = network::calculateIndex(p.square, p.pieceType, p.color);
        EXPECT_EQ(result[index], 1);
    }

    EXPECT_EQ(result[network::INPUT_SIZE], 0);
}
