using System;
using System.Collections.Generic;
using NeuralNetworkLibrary.Examples.BoardGames.Enums;

namespace NeuralNetworkLibrary.Examples.BoardGames.Implementations
{
    /// <summary>
    /// A simple class that represents a tic tac toe match
    /// </summary>
    public sealed class TicTacToe : RandomCompetitiveCrossesGameBase
    {
        /// <summary>
        /// Gets the size of a TicTacToe game board
        /// </summary>
        private const int BoardSize = 3;

        /// <summary>
        /// Returns a new TicTacToe match
        /// </summary>
        /// <param name="firstTurn">Indicates who's playing the first move</param>
        /// <param name="random">The random provider for the opponent</param>
        public TicTacToe(bool firstTurn, Random random) : base(BoardSize, BoardSize, random, firstTurn)
        {
            if (!firstTurn)
            {
                Board[RandomProvider.Next(0, BoardSize), RandomProvider.Next(0, BoardSize)] = GameBoardTileValue.Cross;
                AvailableMoves--;
            }
        }

        #region Implementation

        /// <summary>
        /// Checks the result of the current match, returns Tie if the match isn't finished yet too
        /// </summary>
        public override CrossesGameResult CheckMatchResult()
        {
            // Horizontal crosses
            for (int i = 0; i < 3; i++)
            {
                if (Board[i, 0] == Board[i, 1] && Board[i, 0] == Board[i, 2] && Board[i, 0] != GameBoardTileValue.Empty)
                {
                    return Board[i, 0] == GameBoardTileValue.Nought
                        ? CrossesGameResult.PlayerVictory
                        : CrossesGameResult.OpponentVictory;
                }
            }

            // Vertical crosses
            for (int i = 0; i < 3; i++)
            {
                if (Board[0, i] == Board[1, i] && Board[0, i] == Board[2, i] && Board[0, i] != GameBoardTileValue.Empty)
                {
                    return Board[0, i] == GameBoardTileValue.Nought
                        ? CrossesGameResult.PlayerVictory
                        : CrossesGameResult.OpponentVictory;
                }
            }

            // Diagonal crosses
            if (Board[0, 0] == Board[1, 1] && Board[1, 1] == Board[2, 2] && Board[0, 0] != GameBoardTileValue.Empty)
            {
                return Board[0, 0] == GameBoardTileValue.Nought
                    ? CrossesGameResult.PlayerVictory
                    : CrossesGameResult.OpponentVictory;
            }
            if (Board[0, 2] == Board[1, 1] && Board[1, 1] == Board[2, 0] && Board[1, 1] != GameBoardTileValue.Empty)
            {
                return Board[0, 2] == GameBoardTileValue.Nought
                    ? CrossesGameResult.PlayerVictory
                    : CrossesGameResult.OpponentVictory;
            }

            // No winner
            return CrossesGameResult.Tie;
        }

        /// <summary>
        /// Performs the player move, throws an InvalidOperationException if it's the opponent's turn
        /// </summary>
        /// <param name="x">The target row</param>
        /// <param name="y">The target column</param>
        /// <param name="auto">If true and the target position isn't empty, the first empty tile will be used</param>
        public bool Move(int x, int y, bool auto)
        {
            // Turn check
            if (_PlayerTurn) throw new InvalidOperationException("It is not the plyer's turn");
            if (AvailableMoves == 0) throw new InvalidOperationException("The game is already over");

            // Check if the move is valid
            if (Board[x, y] == GameBoardTileValue.Empty)
            {
                Board[x, y] = GameBoardTileValue.Nought;
                AvailableMoves--;
                _PlayerTurn = false;
                return true;
            }

            // Automatically cross the first empty space if necessary
            if (auto)
            {
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        if (Board[x, y] == 0)
                        {
                            Board[x, y] = GameBoardTileValue.Nought;
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
            List<int[]> free = new List<int[]>();
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    if (Board[i, j] == 0) free.Add(new[] { i, j });
                }
            }

            int[] chosen = free[RandomProvider.Next(0, free.Count)];
            Board[chosen[0], chosen[1]] = GameBoardTileValue.Cross;
            AvailableMoves--;
            _PlayerTurn = true;
        }

        #endregion
    }
}
