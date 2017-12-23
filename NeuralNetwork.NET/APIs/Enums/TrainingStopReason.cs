namespace NeuralNetworkNET.APIs.Enums
{
    /// <summary>
    /// Indicates the reason why a network training session has stopped
    /// </summary>
    public enum TrainingStopReason
    {
        /// <summary>
        /// All the expected training epochs have been completed correctly
        /// </summary>
        EpochsCompleted,

        /// <summary>
        /// The validation test has detected a convergence withing the specified parameters and the training has been halted
        /// </summary>
        EarlyStopping,

        /// <summary>
        /// The training was explicitly stopped before its completion
        /// </summary>
        TrainingCanceled,

        /// <summary>
        /// The weights or biases in the network became invalid during the training
        /// </summary>
        NumericOverflow
    }
}