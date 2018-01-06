namespace NeuralNetworkNET.APIs.Interfaces.Data
{
    /// <summary>
    /// An interface for a batched dataset used to train a network
    /// </summary>
    public interface ITrainingDataset : IDataset
    {
        /// <summary>
        /// Gets or sets the number of samples in each samples batch in the current dataset
        /// </summary>
        int BatchSize { get; set; }
    }
}
