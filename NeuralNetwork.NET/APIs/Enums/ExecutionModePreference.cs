namespace NeuralNetworkNET.APIs.Enums
{
    /// <summary>
    /// Indicates the preferred mode of execution for all expensive operations in the library
    /// </summary>
    public enum ExecutionModePreference
    {
        /// <summary>
        /// Only CPU-powered functions are executed. This means that network layers are deserialized through the <see cref="NetworkLayers"/> class,
        /// and that all other computations will only be scheduled on the CPU.
        /// </summary>
        Cpu,
        
        /// <summary>
        /// CUDA-powered functions are supported. When using this mode, network layers are deserialized through the <see cref="CuDnnNetworkLayers"/> class,
        /// and all other available operations will be scheduled on the GPU.
        /// </summary>
        Cuda
    }
}