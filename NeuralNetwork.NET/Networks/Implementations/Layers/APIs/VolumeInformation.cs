using System;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Implementations.Layers.APIs
{
    /// <summary>
    /// A struct that represents a data volume with square 2D slices
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    public struct VolumeInformation
    {
        /// <summary>
        /// Gets the axis of each 2D slice
        /// </summary>
        [JsonProperty(nameof(Axis), Order = 1)]
        public readonly int Axis;

        /// <summary>
        /// Gets the depth of the data volume
        /// </summary>
        [JsonProperty(nameof(Depth), Order = 2)]
        public readonly int Depth;

        public VolumeInformation(int axis, int depth)
        {
            if (axis <= 0) throw new ArgumentException("The height and width of the kernels must be positive values");
            if (depth <= 0) throw new ArgumentOutOfRangeException(nameof(depth), "The depth of each kernel must be positive");
            Axis = axis;
            Depth = depth;
        }

        /// <summary>
        /// Gets the total number of entries in the data volume
        /// </summary>
        [JsonProperty(nameof(Size), Order = 3)]
        public int Size => Axis * Axis * Depth;
        
        /// <inheritdoc/>
        public override String ToString() => $"Axis: {Axis}, Depth: {Depth}, Size: {Size}";
    }
}
