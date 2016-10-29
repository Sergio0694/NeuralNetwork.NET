using NeuralNetworkLibrary.Examples.CrossesGames.Enums;

namespace NeuralNetworkLibrary.Examples.CrossesGames
{
    /// <summary>
    /// The base abstract class for all the crosses games
    /// </summary>
    public abstract class CrossesGameBase
    {
        #region Fields and parameters

        /// <summary>
        /// Gets the height of the game board
        /// </summary>
        public int Height { get; }

        /// <summary>
        /// Gets the width of the game board
        /// </summary>
        public int Width { get; }

        /// <summary>
        /// Gets the number of tiles on the board
        /// </summary>
        private readonly int TotalTiles;

        /// <summary>
        /// Gets the board game
        /// </summary>
        protected readonly GameBoardTileValue[,] Board;

        /// <summary>
        /// Gets the number of remaining moves
        /// </summary>
        public int AvailableMoves { get; protected set; }

        #endregion

        /// <summary>
        /// Creates a new instance with the given size
        /// </summary>
        /// <param name="height">The height of the game board</param>
        /// <param name="width">The width of the game bard</param>
        protected CrossesGameBase(int height, int width)
        {
            Height = height;
            Width = width;
            Board = new GameBoardTileValue[height, width];
            TotalTiles = height * width;
            AvailableMoves = TotalTiles;
        }

        /// <summary>
        /// Serializes the current game state into a linear 1 * Size matrix
        /// </summary>
        public double[,] Serialize()
        {
            double[,] board = new double[1, TotalTiles];
            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                {
                    board[0, i * Width + j] = (double)Board[i, j];
                }
            }
            return board;
        }
    }
}
