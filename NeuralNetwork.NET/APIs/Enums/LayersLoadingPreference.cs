namespace NeuralNetworkNET.APIs.Enums
{
    /// <summary>
    /// Indicates the preferred type of network layers to serialize, whenever possible
    /// </summary>
    public enum LayersLoadingPreference
    {
        /// <summary>
        /// The network layers are deserialized through the <see cref="NetworkLayers"/> class, if possible
        /// </summary>
        Cpu,
        
        /// <summary>
        /// The network layers are deserialized through the <see cref="CuDnnNetworkLayers"/> class
        /// </summary>
        Cuda
    }
}