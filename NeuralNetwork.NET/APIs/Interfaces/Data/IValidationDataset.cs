namespace NeuralNetworkNET.APIs.Interfaces.Data
{
    /// <summary>
    /// An interface for a validation dataset, with user-defined tolerance parameters
    /// </summary>
    public interface IValidationDataset : IDataset
    {
        /// <summary>
        /// Gets the convergence tolerance for the validation dataset
        /// </summary>
        float Tolerance { get; set; }

        /// <summary>
        /// Gets the maximum number of epochs allowed to pass within the tolerance threshold before stopping the training
        /// </summary>
        int EpochsInterval { get; set; }
    }
}
