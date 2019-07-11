using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetwork.NET.Core.Extensions;
using NeuralNetwork.NET.Core.Helpers;
using SixLabors.ImageSharp.PixelFormats;

namespace NeuralNetwork.NET.Core.Structs.Info
{
    /// <summary>
    /// A <see langword="struct"/> that contains info on the size of a given <see cref="Tensor"/>
    /// </summary>
    [DebuggerDisplay("C: {C}, H: {H}, W: {W}, Size: {VolumeSize}")]
    public readonly struct TensorInfo : IEquatable<TensorInfo>
    {
        #region Fields and parameters

        /// <summary>
        /// The number of channels for the tensor description
        /// </summary>
        public readonly int C;

        /// <summary>
        /// The height of each 2D slice
        /// </summary>
        public readonly int H;

        /// <summary>
        /// The width of each 2D slice
        /// </summary>
        public readonly int W;

        /// <summary>
        /// Gets the total number of entries in the data volume
        /// </summary>
        public int VolumeSize
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => C * H * W;
        }

        /// <summary>
        /// Gets the size of each 2D size
        /// </summary>
        public int SliceSize
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => H * W;
        }

        /// <summary>
        /// Gets whether the current <see cref="Tensor"/> instance is empty
        /// </summary>
        public bool IsEmpty
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => C == 0 || H == 0 || W == 0;
        }

        #endregion

        #region Constructors

        internal TensorInfo(int c, int h, int w)
        {
            Guard.IsTrue(c >= 1, nameof(c), "The C parameter must be a positive number");
            Guard.IsTrue(h >= 1, nameof(h), "The H parameter must be a positive number");
            Guard.IsTrue(w >= 1, nameof(w), "The W parameter must be a positive number");

            C = c;
            H = h;
            W = w;
        }

        /// <summary>
        /// Creates a new <see cref="TensorInfo"/> instance with the specified parameters
        /// </summary>
        /// <param name="w">The size of the <see cref="TensorInfo"/> instance to create</param>
        [Pure]
        public static TensorInfo New(int w) => new TensorInfo(1, 1, w);

        /// <summary>
        /// Creates a new <see cref="TensorInfo"/> instance with the specified parameters
        /// </summary>
        /// <param name="h">The height of the 2D area in the new <see cref="TensorInfo"/> instance</param>
        /// <param name="w">The width of the 2D area in the new <see cref="TensorInfo"/> instance</param>
        [Pure]
        public static TensorInfo New(int h, int w) => new TensorInfo(1, h, w);

        /// <summary>
        /// Creates a new <see cref="TensorInfo"/> instance with the specified parameters
        /// </summary>
        /// <param name="c">The number of channels in the new <see cref="TensorInfo"/> instance</param>
        /// <param name="h">The height of the 2D area in the new <see cref="TensorInfo"/> instance</param>
        /// <param name="w">The width of the 2D area in the new <see cref="TensorInfo"/> instance</param>
        [Pure]
        public static TensorInfo New(int c, int h, int w) => new TensorInfo(c, h, w);

        /// <summary>
        /// Creates a new <see cref="TensorInfo"/> instance for an image with a user-defined pixel type
        /// </summary>
        /// <typeparam name="TPixel">The type of image pixels. It must be either <see cref="Alpha8"/>, <see cref="Rgb24"/> or <see cref="Argb32"/></typeparam>
        /// <param name="h">The height of the images in the new <see cref="TensorInfo"/> instance</param>
        /// <param name="w">The width of the images in the new <see cref="TensorInfo"/> instance</param>
        [PublicAPI]
        [Pure]
        public static TensorInfo New<TPixel>(int h, int w) where TPixel : struct, IPixel<TPixel>
        {
            if (typeof(TPixel) == typeof(Alpha8)) return new TensorInfo(1, h, w);
            if (typeof(TPixel) == typeof(Rgb24)) return new TensorInfo(3, h, w);
            if (typeof(TPixel) == typeof(Argb32) || typeof(TPixel) == typeof(Rgba32)) return new TensorInfo(4, h, w);
            throw new InvalidOperationException($"The {typeof(TPixel).Name} pixel format isn't currently supported");
        }

        #endregion

        #region Equality

        /// <inheritdoc/>
        public bool Equals(TensorInfo other) => C == other.C && H == other.H && W == other.W;

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is TensorInfo tensor && Equals(tensor);

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            Span<int> chw = stackalloc int[] { C, H, W };
            return chw.GetContentHashCode();
        }

        /// <summary>
        /// Checks whether or not two <see cref="TensorInfo"/> instances have the same parameters
        /// </summary>
        /// <param name="a">The first instance</param>
        /// <param name="b">The second instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator ==(in TensorInfo a, in TensorInfo b) => a.Equals(b);

        /// <summary>
        /// Checks whether or not two <see cref="TensorInfo"/> instances have different parameters
        /// </summary>
        /// <param name="a">The first instance</param>
        /// <param name="b">The second instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator !=(in TensorInfo a, in TensorInfo b) => !a.Equals(b);

        #endregion
    }
}
