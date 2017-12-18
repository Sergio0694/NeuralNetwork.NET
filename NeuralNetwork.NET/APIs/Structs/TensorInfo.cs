using JetBrains.Annotations;
using Newtonsoft.Json;
using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworkNET.APIs.Structs
{
    /// <summary>
    /// A <see cref="struct"/> that contains info on the size of a given tensor
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    [DebuggerDisplay("Height: {Height}, Width: {Width}, Channels: {Channels}, Size: {Size}")]
    public readonly struct TensorInfo : IEquatable<TensorInfo>
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
        /// Gets the number of channels for the tensor description
        /// </summary>
        [JsonProperty(nameof(Channels), Order = 3)]
        public readonly int Channels;

        /// <summary>
        /// Gets the total number of entries in the data volume
        /// </summary>
        [JsonProperty(nameof(Size), Order = 3)]
        public int Size
        {
            [Pure]
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Height * Width * Channels;
        }

        /// <summary>
        /// Gets the size of each 2D size
        /// </summary>
        public int SliceSize
        {
            [Pure]
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Height * Width;
        }

        internal TensorInfo(int height, int width, int channels)
        {
            if (height * width <= 0) throw new ArgumentException("The height and width of the kernels must be positive values");
            if (channels < 1) throw new ArgumentOutOfRangeException(nameof(channels), "The number of channels must be at least equal to 1");
            Height = height;
            Width = width;
            Channels = channels;
        }

        /// <summary>
        /// Creates a new <see cref="TensorInfo"/> instance for an RGB image
        /// </summary>
        /// <param name="height">The height of the input image</param>
        /// <param name="width">The width of the input image</param>
        [PublicAPI]
        [Pure]
        public static TensorInfo CreateForRgbImage(int height, int width) => new TensorInfo(height, width, 3);

        /// <summary>
        /// Creates a new <see cref="TensorInfo"/> instance for a grayscale image
        /// </summary>
        /// <param name="height">The height of the input image</param>
        /// <param name="width">The width of the input image</param>
        [PublicAPI]
        [Pure]
        public static TensorInfo CreateForGrayscaleImage(int height, int width) => new TensorInfo(height, width, 1);

        /// <summary>
        /// Creates a new <see cref="TensorInfo"/> instance for a linear network layer, without keeping track of spatial info
        /// </summary>
        /// <param name="size">The input size</param>
        [PublicAPI]
        [Pure]
        public static TensorInfo CreateLinear(int size) => new TensorInfo(1, 1, size);

        #region Equality

        /// <inheritdoc/>
        public bool Equals(TensorInfo other) => this == other;

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is TensorInfo tensor ? this == tensor : false;

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            int hash = 17;
            unchecked
            {
                hash += Height;
                hash = hash * 23 + Width;
                hash = hash * 23 + Channels;
            }
            return hash;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator ==(in TensorInfo a, in TensorInfo b) => a.Height == b.Height && a.Width == b.Width && a.Channels == b.Channels;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator !=(in TensorInfo a, in TensorInfo b) => !(a == b);

        #endregion
    }
}
