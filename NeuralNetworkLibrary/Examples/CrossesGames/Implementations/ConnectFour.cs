using System;
using System.Collections.Generic;
using NeuralNetworkLibrary.Examples.CrossesGames.Enums;

namespace NeuralNetworkLibrary.Examples.CrossesGames.Implementations
{
    /// <summary>
    /// A simple class that represents the classic Force4 game
    /// </summary>
    public sealed class ConnectFour : RandomCompetitiveCrossesGameBase
    {
        /// <summary>
        /// Gets the height of the game board
        /// </summary>
        private const int BoardHeight = 6;

        /// <summary>
        /// Gets the width of the game board
        /// </summary>
        private const int BoardWidth = 7;

        /// <summary>
        /// Returns a new Force4 match
        /// </summary>
        /// <param name="firstTurn">Indicates who's playing the first move</param>
        /// <param name="random">The random provider for the opponent</param>
        public ConnectFour(bool firstTurn, Random random) : base(BoardHeight, BoardWidth, random, firstTurn)
        {
            if (!firstTurn)
            {
                MoveOpponent();
            }
        }

        #region Implementation

        /// <summary>
        /// Gets the bitboard for the player
        /// </summary>
        private ulong PlayerBitboard;

        /// <summary>
        /// Gets the bitboard for the opponent
        /// </summary>
        private ulong OpponentBitboard;

        /// <summary>
        /// Calculates a new bitboard with the new move
        /// </summary>
        /// <param name="board">The source bitboard</param>
        /// <param name="x">The target row</param>
        /// <param name="y">The target column</param>
        private ulong SetMove(ulong board, int x, int y)
        {
            int index = BoardHeight * x + y;
            return board | (1UL << index);
        }

        /// <summary>
        /// Checks if the given bitboard has won
        /// </summary>
        /// <param name="board">The bitboard to check</param>
        private bool CheckBitboardWin(ulong board)
        {
            ulong y = board & (board >> 6);
            if ((y & (y >> 2 * 6)) > 0) return true;
            y = board & (board >> 7);
            if ((y & (y >> 2 * 7)) > 0) return true;
            y = board & (board >> 8);
            if ((y & (y >> 2 * 8))  > 0) return true;
            y = board & (board >> 1);
            return (y & (y >> 2)) > 0;
        }

        /// <summary>
        /// Checks the result of the current match, returns Tie if the match isn't finished yet too
        /// </summary>
        public override CrossesGameResult CheckMatchResult()
        {
            // Check player and opponent
            if (_PlayerTurn && CheckBitboardWin(OpponentBitboard)) return CrossesGameResult.OpponentVictory;
            if (!_PlayerTurn && CheckBitboardWin(PlayerBitboard)) return CrossesGameResult.PlayerVictory;

            // No winner
            return CrossesGameResult.Tie;
        }

        /// <summary>
        /// Performs the player move, throws an InvalidOperationException if it's the opponent's turn
        /// </summary>
        /// <param name="index">The target column</param>
        /// <param name="auto">If true and the target position isn't empty, the first empty tile will be used</param>
        public bool Move(int index, bool auto)
        {
            // Turn check
            if (_PlayerTurn) throw new InvalidOperationException("It is not the plyer's turn");
            if (AvailableMoves == 0) throw new InvalidOperationException("The game is already over");

            // Check if the move is valid
            for (int i = BoardHeight - 1; i >= 0; i--)
            {
                if (Board[i, index] == GameBoardTileValue.Empty)
                {
                    Board[i, index] = GameBoardTileValue.Nought;
                    PlayerBitboard = SetMove(PlayerBitboard, i, index);
                    AvailableMoves--;
                    _PlayerTurn = false;
                    return true;
                }
            }

            // Automatically cross the first empty space if necessary
            if (auto)
            {
                for (int i = 0; i < 3; i++)
                {
                    for (int j = BoardHeight - 1; j >= 0; j--)
                    {
                        if (Board[j, i] == GameBoardTileValue.Empty)
                        {
                            Board[j, i] = GameBoardTileValue.Nought;
                            PlayerBitboard = SetMove(PlayerBitboard, i, index);
                            AvailableMoves--;
                            _PlayerTurn = false;
                            return false;
                        }
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// Performs the random move for the opponent
        /// </summary>
        public override void MoveOpponent()
        {
            // Find the free positions
            if (_PlayerTurn || AvailableMoves == 0) throw new InvalidOperationException();
            List<int> free = new List<int>();
            for (int i = 0; i < 3; i++)
            {
                for (int j = BoardHeight - 1; j >= 0; j--)
                {
                    if (Board[j, i] == GameBoardTileValue.Empty)
                    {
                        free.Add(i);
                        break;
                    }
                }
            }

            // Perform the random move
            int chosen = free[RandomProvider.Next(0, free.Count)];
            for (int j = BoardHeight - 1; j >= 0; j--)
            {
                if (Board[j, chosen] == GameBoardTileValue.Empty)
                {
                    Board[j, chosen] = GameBoardTileValue.Cross;
                    OpponentBitboard = SetMove(OpponentBitboard, j, chosen);
                    break;
                }
            }
            AvailableMoves--;
        }

        #endregion
    }
}
