namespace NeuralNetworkDotNet.Core.Enums
{
    /// <summary>
    /// An <see langword="enum"/> that indicates a specific allocation mode for memory buffers
    /// </summary>
    public enum AllocationMode
    {
        /// <summary>
        /// The default allocation mode, with no guarantees on the initial data in the allocated buffers
        /// </summary>
        Default,

        /// <summary>
        /// Clean allocation mode, where the allocated buffers are initially zeroed
        /// </summary>
        Clean
    }
}
