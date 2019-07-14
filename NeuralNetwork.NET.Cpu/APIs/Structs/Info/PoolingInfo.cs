using System;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.Helpers;

namespace NeuralNetworkDotNet.APIs.Structs.Info
{
    /// <summary>
    /// A <see langword="struct"/> containing all the info on a pooling operation
    /// </summary>
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

        // Private constructor
        private PoolingInfo(
            PoolingMode mode, int windowHeight, int windowWidth,
            int verticalPadding, int horizontalPadding,
            int verticalStride, int horizontalStride)
        {
            Guard.IsTrue(windowHeight > 0, nameof(windowHeight), "The window height must be at least equal to 1");
            Guard.IsTrue(windowWidth > 0, nameof(windowWidth), "The window width must be at least equal to 1");
            Guard.IsTrue(verticalPadding >= 0, nameof(verticalPadding), "The vertical padding must be greater than or equal to 0");
            Guard.IsTrue(horizontalPadding >= 0, nameof(horizontalPadding), "The horizontal padding must be greater than or equal to 0");
            Guard.IsTrue(verticalStride >= 1, nameof(verticalStride), "The vertical stride must be at least equal to 1");
            Guard.IsTrue(horizontalStride >= 1, nameof(horizontalStride), "The horizontal stride must be at least equal to 1");

            WindowHeight = windowHeight;
            WindowWidth = windowWidth;
            VerticalPadding = verticalPadding;
            HorizontalPadding = horizontalPadding;
            VerticalStride = verticalStride;
            HorizontalStride = horizontalStride;
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
        internal Shape GetOutputShape(Shape input)
        {
            int
                h = (input.H - WindowHeight + 2 * VerticalPadding) / VerticalStride + 1,
                w = (input.W - WindowWidth + 2 * HorizontalPadding) / HorizontalStride + 1;

            Guard.IsTrue(h > 0 && w > 0, "The input tensor shape is not valid to apply the current pooling operation");

            return (input.C, h, w);
        }

        #region IEquatable<PoolingInfo>

        /// <inheritdoc/>
        public bool Equals(PoolingInfo other) => this == other;

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is PoolingInfo info && this == info;

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            Span<int> values = stackalloc int[] { (int)Mode, WindowHeight, WindowWidth, VerticalPadding, HorizontalPadding, VerticalStride, HorizontalStride };
            return values.GetContentHashCode();
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
