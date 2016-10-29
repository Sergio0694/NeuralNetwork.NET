using System;
using NeuralNetworkLibrary.Examples.BoardGames.Enums;

namespace NeuralNetworkLibrary.Examples.BoardGames
{
    /// <summary>
    /// Base class for a game played against a random opponent
    /// </summary>
    public abstract class RandomCompetitiveCrossesGameBase : BoardGameBase<GameBoardTileValue>
    {
        #region Fields and parameters

        /// <summary>
        /// Gets the private random provider to control the opponent
        /// </summary>
        protected readonly Random RandomProvider;

        /// <summary>
        /// Gets whether or not it is the player's turn
        /// </summary>
        protected bool _PlayerTurn;

        /// <summary>
        /// Gets the number of remaining moves
        /// </summary>
        public int AvailableMoves { get; protected set; }

        #endregion

        /// <summary>
        /// Creates a new instance of the game
        /// </summary>
        /// <param name="height">Height of the game board</param>
        /// <param name="width">Width of the game board</param>
        /// <param name="random">Random provider to generate the moves of the opponent</param>
        /// <param name="firstTurn">Indicates whether or not the player will move first</param>
        protected RandomCompetitiveCrossesGameBase(int height, int width, Random random, bool firstTurn) : base(height, width)
        {
            // Fixed fields
            RandomProvider = random;
            AvailableMoves = height * width;

            // First turn
            if (firstTurn) _PlayerTurn = true;
        }

        // Serialization converter
        protected override double this[int x, int y] => (double)Board[x, y];

        #region Abstract methods

        /// <summary>
        /// Performs the random move for the opponent
        /// </summary>
        public abstract void MoveOpponent();

        /// <summary>
        /// Checks the result of the current match, returns Tie if the match isn't finished yet too
        /// </summary>
        public abstract CrossesGameResult CheckMatchResult();

        #endregion
    }
}
