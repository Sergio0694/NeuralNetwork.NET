using System;
using System.Diagnostics;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Implementations.Layers.APIs
{
    /// <summary>
    /// A struct that represents a data volume with square 2D slices
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    [DebuggerDisplay("Height: {Height}, Width: {Width}, Depth: {Depth}, Size: {Volume}")]
    public struct VolumeInformation
    {
        /// <summary>
        /// Gets the height of each 2D slice
        /// </summary>
        [JsonProperty(nameof(Height), Order = 1)]
        public readonly int Height;

        /// <summary>
        /// Gets the width of each 2D slice
        /// </summary>
        [JsonProperty(nameof(Width), Order = 2)]
        public readonly int Width;

        /// <summary>
        /// Gets the depth of the data volume
        /// </summary>
        [JsonProperty(nameof(Depth), Order = 3)]
        public readonly int Depth;

        public VolumeInformation(int height, int width, int depth)
        {
            if (height * width <= 0) throw new ArgumentException("The height and width of the kernels must be positive values");
            if (depth < 1) throw new ArgumentOutOfRangeException(nameof(depth), "The depth of each kernel must be positive");
            Height = height;
            Width = width;
            Depth = depth;
        }

        /// <summary>
        /// Gets the total number of entries in the data volume
        /// </summary>
        [JsonProperty(nameof(Volume), Order = 3)]
        public int Volume => Height * Width * Depth;

        /// <summary>
        /// Gets the size of each 2D size
        /// </summary>
        public int SliceSize => Height * Width;

        // Implicit converter
        public static implicit operator VolumeInformation((int X, int Y, int Depth) size) => new VolumeInformation(size.X, size.Y, size.Depth);
    }
}
