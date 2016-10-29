namespace NeuralNetworkLibrary.Examples.BoardGames
{
    /// <summary>
    /// The base abstract class for all the crosses games
    /// </summary>
    public abstract class BoardGameBase<T>
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
        protected readonly T[,] Board;

        #endregion

        /// <summary>
        /// Creates a new instance with the given size
        /// </summary>
        /// <param name="height">The height of the game board</param>
        /// <param name="width">The width of the game bard</param>
        protected BoardGameBase(int height, int width)
        {
            Height = height;
            Width = width;
            Board = new T[height, width];
            TotalTiles = height * width;
        }

        /// <summary>
        /// The access method to get the serialized values for the game board
        /// </summary>
        /// <param name="x">The target row</param>
        /// <param name="y">The target column</param>
        protected abstract double this[int x, int y] { get; }

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
                    board[0, i * Width + j] = this[i, j];
                }
            }
            return board;
        }
    }
}
