namespace NeuralNetworkLibrary.Examples.BoardGames.Enums
{
    /// <summary>
    /// Indicates the value of a given position on the game board
    /// </summary>
    public enum GameBoardTileValue
    {
        /// <summary>
        /// An empty tile
        /// </summary>
        Empty = 0,

        /// <summary>
        /// First player value
        /// </summary>
        Nought = 1,

        /// <summary>
        /// Second player or opponent value
        /// </summary>
        Cross = -1
    }
}