using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using Newtonsoft.Json;
using System;
using System.Runtime.CompilerServices;

namespace NeuralNetworkNET.APIs.Structs
{
    /// <summary>
    /// A <see langword="struct"/> containing all the info on a pooling operation
    /// </summary>
    [JsonObject(MemberSerialization.Fields)]
    public readonly struct PoolingInfo : IEquatable<PoolingInfo>
    {
        /// <summary>
        /// The current pooling mode for the layer
        /// </summary>
        public readonly PoolingMode Mode;

        /// <summary>
        /// The height of each input local receptive field
        /// </summary>
        public readonly int WindowHeight;

        /// <summary>
        /// The width of each input local receptive field
        /// </summary>
        public readonly int WindowWidth;

        /// <summary>
        /// The optional vertical padding for the pooling operation
        /// </summary>
        public readonly int VerticalPadding;

        /// <summary>
        /// The optional horizontal padding for the pooling operation
        /// </summary>
        public readonly int HorizontalPadding;

        /// <summary>
        /// The vertical stride length while sliding the receptive window over the input
        /// </summary>
        public readonly int VerticalStride;

        /// <summary>
        /// The horizontal stride length while sliding the receptive window over the input
        /// </summary>
        public readonly int HorizontalStride;

        #region Constructors

        // Internal constructor
        private PoolingInfo(
            PoolingMode mode, int windowHeight, int windowWidth,
            int verticalPadding, int horizontalPadding,
            int verticalStride, int horizontalStride)
        {
            WindowHeight = windowHeight > 0 ? windowHeight : throw new ArgumentOutOfRangeException(nameof(windowHeight), "The window height must be at least equal to 1");
            WindowWidth = windowWidth > 0 ? windowWidth : throw new ArgumentOutOfRangeException(nameof(windowWidth), "The window width must be at least equal to 1");
            VerticalPadding = verticalPadding >= 0 ? verticalPadding : throw new ArgumentOutOfRangeException(nameof(verticalPadding), "The vertical padding must be greater than or equal to 0");
            HorizontalPadding = horizontalPadding >= 0 ? horizontalPadding : throw new ArgumentOutOfRangeException(nameof(horizontalPadding), "The horizontal padding must be greater than or equal to 0");
            VerticalStride = verticalStride >= 1 ? verticalStride : throw new ArgumentOutOfRangeException(nameof(verticalStride), "The vertical stride must be at least equal to 1");
            HorizontalStride = horizontalStride >= 1 ? horizontalStride : throw new ArgumentOutOfRangeException(nameof(horizontalStride), "The horizontal stride must be at least equal to 1");
            Mode = mode;
        }

        /// <summary>
        /// Gets the default 2x2 max pooling info, with a stride of 2 in both directions and no padding
        /// </summary>
        public static PoolingInfo Default { get; } = new PoolingInfo(PoolingMode.Max, 2, 2, 0, 0, 2, 2);

        /// <summary>
        /// Creates a new pooling operation description with the input parameters
        /// </summary>
        /// <param name="mode">The desired pooling mode to use</param>
        /// <param name="windowHeight">The pooling window height to use</param>
        /// <param name="windowWidth">The pooling window width to use</param>
        /// <param name="verticalPadding">The optional pooling vertical padding</param>
        /// <param name="horizontalPadding">The optional pooling horizontal padding</param>
        /// <param name="verticalStride">The pooling vertical stride size</param>
        /// <param name="horizontalStride">The pooling horizontal stride size</param>
        [PublicAPI]
        [Pure]
        public static PoolingInfo New(
            PoolingMode mode, int windowHeight = 2, int windowWidth = 2,
            int verticalPadding = 0, int horizontalPadding = 0,
            int verticalStride = 2, int horizontalStride = 2)
            => new PoolingInfo(mode, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);

        #endregion

        /// <summary>
        /// Calculates the output size after applying a pooling operation to the input tensor
        /// </summary>
        /// <param name="input">The info on the input tensor</param>
        [Pure]
        internal TensorInfo GetForwardOutputTensorInfo(in TensorInfo input)
        {
            int
                h = (input.Height - WindowHeight + 2 * VerticalPadding) / VerticalStride + 1,
                w = (input.Width - WindowWidth + 2 * HorizontalPadding) / HorizontalStride + 1;
            if (h <= 0 || w <= 0) throw new InvalidOperationException("The input tensor shape is not valid to apply the current pooling operation");
            return new TensorInfo(h, w, input.Channels);
        }

        #region Equality

        /// <inheritdoc/>
        public bool Equals(PoolingInfo other) => this == other;

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is PoolingInfo info && this == info;

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            int hash = 17;
            unchecked
            {
                hash = hash * 31 + (int)Mode;
                hash = hash * 31 + WindowHeight;
                hash = hash * 31 + WindowWidth;
                hash = hash * 31 + VerticalPadding;
                hash = hash * 31 + HorizontalPadding;
                hash = hash * 31 + VerticalStride;
                hash = hash * 31 + HorizontalStride;
            }
            return hash;
        }

        /// <summary>
        /// Checks whether or not two <see cref="PoolingInfo"/> instances have the same parameters
        /// </summary>
        /// <param name="a">The first instance</param>
        /// <param name="b">The second instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator ==(in PoolingInfo a, in PoolingInfo b) => a.Mode == b.Mode &&
                                                                              a.WindowHeight == b.WindowHeight && a.WindowWidth == b.WindowWidth &&
                                                                              a.VerticalPadding == b.VerticalPadding && a.HorizontalPadding == b.HorizontalPadding &&
                                                                              a.VerticalStride == b.VerticalStride && a.HorizontalStride == b.HorizontalStride;

        /// <summary>
        /// Checks whether or not two <see cref="PoolingInfo"/> instances have different parameters
        /// </summary>
        /// <param name="a">The first instance</param>
        /// <param name="b">The second instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator !=(in PoolingInfo a, in PoolingInfo b) => !(a == b);

        #endregion
    }
}
