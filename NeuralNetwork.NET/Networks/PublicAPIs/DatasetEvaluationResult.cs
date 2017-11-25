namespace NeuralNetworkNET.Networks.PublicAPIs
{
    /// <summary>
    /// A struct that contains information on a single completed evaluation of a test dataset
    /// </summary>
    public struct DatasetEvaluationResult
    {
        /// <summary>
        /// Gets the total cost for the evaluated dataset
        /// </summary>
        public float Cost { get; }

        /// <summary>
        /// Gets the classification accuracy for the evaluated dataset
        /// </summary>
        public float Accuracy { get; }

        // Internal constructor
        internal DatasetEvaluationResult(float cost, float accuracy)
        {
            Cost = cost;
            Accuracy = accuracy;
        }
    }
}