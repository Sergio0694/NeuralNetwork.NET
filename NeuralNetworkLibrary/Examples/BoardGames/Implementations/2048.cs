using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworkLibrary.Examples.BoardGames.Enums;

namespace NeuralNetworkLibrary.Examples.BoardGames.Implementations
{
    /// <summary>
    /// A class that represents a 2048 game
    /// </summary>
    public sealed class _2048 : BoardGameBase<int>
    {
        #region Private fields and parameters

        /// <summary>
        /// Gets the size of the game board
        /// </summary>
        private const int BoardSize = 4;

        /// <summary>
        /// Gets the random provider used to spawn new numbers on the board
        /// </summary>
        private readonly Random RandomProvider;

        /// <summary>
        /// Gets the next random number that must be placed on the board
        /// </summary>
        private int NextRandomTile => RandomProvider.Next(0, 10) == 0 ? 4 : 2;

        #endregion

        /// <summary>
        /// Creates a new game with the given random provider
        /// </summary>
        /// <param name="random">The random provider to use during the game</param>
        public _2048(Random random) : base(BoardSize, BoardSize)
        {
            // Store the random provider
            RandomProvider = random;

            // Spawn the first two numbers on the board
            int x1, y1, x2, y2;
            do
            {
                x1 = random.Next(0, BoardSize);
                y1 = random.Next(0, BoardSize);
                x2 = random.Next(0, BoardSize);
                y2 = random.Next(0, BoardSize);
            } while (x1 == x2 && y1 == y2);
            Board[x1, y1] = NextRandomTile;
            Board[x2, y2] = NextRandomTile;
            Score = 4;
            MaxTile = 2;
        }

        /// <summary>
        /// Gets the actual score for the game
        /// </summary>
        public int Score { get; private set; }

        /// <summary>
        /// Gets the maximum tile value reached in the game
        /// </summary>
        public int MaxTile { get; private set; }

        #region Implementation

        // Protected access method
        protected override double this[int x, int y] => Board[x, y];

        /// <summary>
        /// Executes the player move in the given direction
        /// </summary>
        /// <param name="direction">The direction for the next move</param>
        /// <param name="checkOnly">Indicates whether or not to actually execute the move or just check if it's valid</param>
        public bool Move(Direction direction, bool checkOnly)
        {
            int x, y;
            bool valid = false, merged;

            // Vertical directions
            if (direction == Direction.Up || direction == Direction.Down)
            {
                for (int j = 0; j < BoardSize; j++)
                {
                    int ultimo_x = -1;
                    if (direction == Direction.Up)
                    {
                        // Move from bottom to top
                        for (int i = 1; i < BoardSize; i++)
                        {
                            // Move if the current place isn't empty
                            merged = false;
                            if (Board[i, j] != 0)
                            {
                                x = i;
                                y = j;
                                while (x > 0)
                                {
                                    // Previous place empty
                                    if (Board[x - 1, y] == 0)
                                    {
                                        if (checkOnly) return true;
                                        Board[x - 1, y] = Board[x, y];
                                        Board[x, y] = 0;
                                        x -= 1;
                                        valid = true;
                                    }
                                    else if (Board[x - 1, y] == Board[x, y] && // Same number on the previous place
                                             !merged && // Not currently in a combo
                                             (x - 1 != ultimo_x)) // The previous tile hasn't been merged in this same move
                                    {
                                        if (checkOnly) return true;
                                        int points = 2 * Board[x, y];
                                        if (points > MaxTile) MaxTile = points;
                                        Board[x - 1, y] = points;
                                        Score += points;
                                        Board[x, y] = 0;
                                        x--;
                                        ultimo_x = x;
                                        merged = true;
                                        valid = true;
                                    }
                                    else break;
                                }
                            }
                        }
                    }
                    else if (direction == Direction.Down)
                    {
                        for (int i = BoardSize - 1; i >= 0; i--)
                        {
                            merged = false;
                            if (Board[i, j] != 0)
                            {
                                x = i;
                                y = j;
                                while (x < BoardSize - 1)
                                {
                                    if (Board[x + 1, y] == 0)
                                    {
                                        if (checkOnly) return true;
                                        Board[x + 1, y] = Board[x, y];
                                        Board[x, y] = 0;
                                        x += 1;
                                        valid = true;
                                    }
                                    else if (Board[x + 1, y] == Board[x, y]
                                             && !merged && (x + 1 != ultimo_x))
                                    {
                                        if (checkOnly) return true;
                                        int points = 2 * Board[x, y];
                                        if (points > MaxTile) MaxTile = points;
                                        Board[x + 1, y] = points;
                                        Score += points;
                                        Board[x, y] = 0;
                                        x++;
                                        ultimo_x = x;
                                        merged = true;
                                        valid = true;
                                    }
                                    else break;
                                }
                            }
                        }
                    }
                    else throw new InvalidOperationException();
                }
            }
            else
            {
                for (int i = 0; i < BoardSize; i++)
                {
                    int ultimo_y = -1;
                    if (direction == Direction.Right)
                    {
                        for (int j = BoardSize - 2; j >= 0; j--)
                        {
                            merged = false;
                            if (Board[i, j] != 0)
                            {
                                x = i;
                                y = j;
                                while (y < BoardSize - 1)
                                {
                                    if (Board[x, y + 1] == 0)
                                    {
                                        if (checkOnly) return true;
                                        Board[x, y + 1] = Board[x, y];
                                        Board[x, y] = 0;
                                        y++;
                                        valid = true;
                                    }
                                    else if (Board[x, y + 1] == Board[x, y]
                                             && !merged && (y + 1 != ultimo_y))
                                    {
                                        if (checkOnly) return true;
                                        int points = 2 * Board[x, y];
                                        if (points > MaxTile) MaxTile = points;
                                        Board[x, y + 1] = points;
                                        Board[x, y] = 0;
                                        y++;
                                        ultimo_y = y;
                                        merged = true;
                                        Score += points;
                                        valid = true;
                                    }
                                    else break;
                                }
                            }
                        }
                    }
                    else if (direction == Direction.Left)
                    {
                        for (int j = 1; j <= BoardSize - 1; j++)
                        {
                            merged = false;
                            if (Board[i, j] != 0)
                            {
                                x = i;
                                y = j;
                                while (y > 0)
                                {
                                    if (Board[x, y - 1] == 0)
                                    {
                                        if (checkOnly) return true;
                                        Board[x, y - 1] = Board[x, y];
                                        Board[x, y] = 0;
                                        y--;
                                        valid = true;
                                    }
                                    else if (Board[x, y - 1] == Board[x, y]
                                              && !merged && (y - 1 != ultimo_y))
                                    {
                                        if (checkOnly) return true;
                                        int points = 2 * Board[x, y];
                                        if (points > MaxTile) MaxTile = points;
                                        Board[x, y - 1] = points;
                                        Board[x, y] = 0;
                                        y--;
                                        ultimo_y = y;
                                        merged = true;
                                        Score += points;
                                        valid = true;
                                    }
                                    else break;
                                }
                            }
                        }
                    }
                    else throw new InvalidOperationException();
                }
            }

            // Add a new random number if necessary
            if (valid)
            {
                List<Tuple<int, int>> free = new List<Tuple<int, int>>();
                for (int i = 0; i < BoardSize; i++)
                {
                    for (int j = 0; j < BoardSize; j++)
                    {
                        if (Board[i, j] == 0) free.Add(Tuple.Create(i, j));
                    }
                }
                if (free.Count > 0)
                {
                    Tuple<int, int> pick = free[RandomProvider.Next(0, free.Count)];
                    Board[pick.Item1, pick.Item2] = NextRandomTile;
                }
            }
            return valid;
        }

        /// <summary>
        /// Gets the next available move
        /// </summary>
        public Direction? NextAvailableMove
        {
            get
            {
                // Check all the possible directions, return null if it is game over
                foreach (Direction direction in new[] { Direction.Up, Direction.Down, Direction.Left, Direction.Right }.Where(direction => Move(direction, true)))
                {
                    return direction;
                }
                return null;
            }
        }

        #endregion
    }
}
