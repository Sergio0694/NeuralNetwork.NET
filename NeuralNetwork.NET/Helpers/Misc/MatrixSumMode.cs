namespace NeuralNetworkNET.Helpers.Misc
{
    /// <summary>
    /// Indicates the mode to sum two matrices together
    /// </summary>
    public enum MatrixSumMode
    {
        /// <summary>
        /// Elementwise sum, where both matrices have the same size
        /// </summary>
        Elementwise,

        /// <summary>
        /// Column-wise sum, where the second matrix is a column vector
        /// </summary>
        ColumnByColumn
    }
}