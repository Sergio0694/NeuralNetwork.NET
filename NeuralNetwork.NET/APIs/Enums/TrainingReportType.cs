namespace NeuralNetworkNET.APIs.Enums
{
    /// <summary>
    /// Indicates a type of training progress report
    /// </summary>
    public enum TrainingReportType
    {
        /// <summary>
        /// The dataset accuracy according to the expected outputs
        /// </summary>
        Accuracy,

        /// <summary>
        /// The cost function value for the current dataset being evaluated
        /// </summary>
        Cost
    }
}
