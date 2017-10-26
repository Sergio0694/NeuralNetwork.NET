namespace NeuralNetworkNET.Cuda.APIs
{
    /// <summary>
    /// Indicates the desired processing mode while training a network
    /// </summary>
    public enum ProcessingMode
    {
        /// <summary>
        /// Perform the processing on the CPU, using the Task Parallel Library
        /// </summary>
        Cpu,

        /// <summary>
        /// Perform the processing on the GPU
        /// </summary>
        Gpu
    }
}