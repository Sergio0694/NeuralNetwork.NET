using JetBrains.Annotations;
using Newtonsoft.Json;
using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace NeuralNetworkNET.APIs.Structs
{
    /// <summary>
    /// A <see langword="struct"/> that contains info on the size of a given tensor
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    [DebuggerDisplay("Height: {Height}, Width: {Width}, Channels: {Channels}, Size: {Size}")]
    public readonly struct TensorInfo : IEquatable<TensorInfo>
    {
        #region Fields and parameters

        /// <summary>
        /// The height of each 2D slice
        /// </summary>
        [JsonProperty(nameof(Height), Order = 1)]
        public readonly int Height;

        /// <summary>
        /// The width of each 2D slice
        /// </summary>
        [JsonProperty(nameof(Width), Order = 2)]
        public readonly int Width;

        /// <summary>
        /// The number of channels for the tensor description
        /// </summary>
        [JsonProperty(nameof(Channels), Order = 3)]
        public readonly int Channels;

        /// <summary>
        /// Gets the total number of entries in the data volume
        /// </summary>
        [JsonProperty(nameof(Size), Order = 4)]
        public int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Height * Width * Channels;
        }

        /// <summary>
        /// Gets the size of each 2D size
        /// </summary>
        public int SliceSize
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Height * Width;
        }

        /// <summary>
        /// Gets whether the current <see cref="Tensor"/> instance is invalid (empty or with invalid parameters)
        /// </summary>
        public bool IsEmptyOrInvalid
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Height <= 0 || Width <= 0 || Channels <= 0;
        }

        #endregion

        #region Constructors

        internal TensorInfo(int height, int width, int channels)
        {
            if (height <= 0 || width <= 0) throw new ArgumentException("The height and width of the kernels must be positive values");
            if (channels <= 0) throw new ArgumentException("The number of channels must be positive");
            Height = height;
            Width = width;
            Channels = channels >= 1 ? channels :  throw new ArgumentOutOfRangeException(nameof(channels), "The number of channels must be at least equal to 1");
        }

        /// <summary>
        /// Creates a new <see cref="TensorInfo"/> instance for a linear network layer, without keeping track of spatial info
        /// </summary>
        /// <param name="size">The input size</param>
        [PublicAPI]
        [Pure]
        public static TensorInfo Linear(int size) => new TensorInfo(1, 1, size);

        /// <summary>
        /// Creates a new <see cref="TensorInfo"/> instance for an image with a user-defined pixel type
        /// </summary>
        /// <typeparam name="TPixel">The type of image pixels. It must be either <see cref="Alpha8"/>, <see cref="Rgb24"/> or <see cref="Argb32"/></typeparam>
        /// <param name="height">The height of the input image</param>
        /// <param name="width">The width of the input image</param>
        [PublicAPI]
        [Pure]
        public static TensorInfo Image<TPixel>(int height, int width) where TPixel : struct, IPixel<TPixel>
        {
            if (typeof(TPixel) == typeof(Alpha8)) return new TensorInfo(height, width, 1);
            if (typeof(TPixel) == typeof(Rgb24)) return new TensorInfo(height, width, 3);
            if (typeof(TPixel) == typeof(Argb32) || typeof(TPixel) == typeof(Rgba32)) return new TensorInfo(height, width, 4);
            throw new InvalidOperationException($"The {typeof(TPixel).Name} pixel format isn't currently supported");
        }

        /// <summary>
        /// Creates a new <see cref="TensorInfo"/> instance for with a custom 3D shape
        /// </summary>
        /// <param name="height">The input volume height</param>
        /// <param name="width">The input volume width</param>
        /// <param name="channels">The number of channels in the input volume</param>
        [PublicAPI]
        [Pure]
        public static TensorInfo Volume(int height, int width, int channels) => new TensorInfo(height, width, channels);

        #endregion

        #region Equality

        /// <inheritdoc/>
        public bool Equals(TensorInfo other) => this == other;

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is TensorInfo tensor && this == tensor;

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

        /// <summary>
        /// Checks whether or not two <see cref="TensorInfo"/> instances have the same parameters
        /// </summary>
        /// <param name="a">The first instance</param>
        /// <param name="b">The second instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator ==(in TensorInfo a, in TensorInfo b) => a.Height == b.Height && a.Width == b.Width && a.Channels == b.Channels;

        /// <summary>
        /// Checks whether or not two <see cref="TensorInfo"/> instances have different parameters
        /// </summary>
        /// <param name="a">The first instance</param>
        /// <param name="b">The second instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator !=(in TensorInfo a, in TensorInfo b) => !(a == b);

        #endregion
    }
}
