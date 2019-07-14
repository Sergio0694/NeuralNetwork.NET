using System;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.Helpers;

namespace NeuralNetworkDotNet.APIs.Structs.Info
{
    /// <summary>
    /// A <see langword="struct"/> containing all the info on a convolution operation
    /// </summary>
    public readonly struct ConvolutionInfo : IEquatable<ConvolutionInfo>
    {
        /// <summary>
        /// The current convolution mode for the operation
        /// </summary>
        public readonly ConvolutionMode Mode;

        /// <summary>
        /// The optional vertical padding for the convolution operation
        /// </summary>
        public readonly int VerticalPadding;

        /// <summary>
        /// The optional horizontal padding for the convolution operation
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

        // Private constructor
        private ConvolutionInfo(
            ConvolutionMode mode,
            int verticalPadding, int horizontalPadding,
            int verticalStride, int horizontalStride)
        {
            Guard.IsTrue(verticalPadding >= 0, nameof(verticalPadding), "The vertical padding must be greater than or equal to 0");
            Guard.IsTrue(horizontalPadding >= 0, nameof(horizontalPadding), "The horizontal padding must be greater than or equal to 0");
            Guard.IsTrue(verticalStride >= 1, nameof(verticalStride), "The vertical stride must be at least equal to 1");
            Guard.IsTrue(horizontalStride >= 1, nameof(horizontalStride), "The horizontal stride must be at least equal to 1");

            Mode = mode;
            VerticalPadding = verticalPadding;
            HorizontalPadding = horizontalPadding;
            VerticalStride = verticalStride;
            HorizontalStride = horizontalStride;
        }

        /// <summary>
        /// Gets the default convolution info, with no padding and a stride of 1 in both directions
        /// </summary>
        public static ConvolutionInfo Default { get; } = new ConvolutionInfo(ConvolutionMode.Convolution, 0, 0, 1, 1);

        /// <summary>
        /// Gets the default cross correlation mode, with no padding and a stride of 1 in both directions
        /// </summary>
        public static ConvolutionInfo CrossCorrelation { get; } = new ConvolutionInfo(ConvolutionMode.CrossCorrelation, 0, 0, 1, 1);

        /// <summary>
        /// Creates a new convolution operation description with the input parameters
        /// </summary>
        /// <param name="mode">The desired convolution mode to use</param>
        /// <param name="verticalPadding">The optional convolution vertical padding</param>
        /// <param name="horizontalPadding">The optional convolution horizontal padding</param>
        /// <param name="verticalStride">The convolution vertical stride size</param>
        /// <param name="horizontalStride">The convolution horizontal stride size</param>
        [PublicAPI]
        [Pure]
        public static ConvolutionInfo New(
            ConvolutionMode mode = ConvolutionMode.Convolution,
            int verticalPadding = 0, int horizontalPadding = 0,
            int verticalStride = 1, int horizontalStride = 1)
            => new ConvolutionInfo(mode, verticalPadding, horizontalPadding, verticalStride, horizontalStride);

        /// <summary>
        /// Calculates the output size after applying a convolution operation to the input tensor
        /// </summary>
        /// <param name="input">The info on the input tensor</param>
        /// <param name="size">The size of the convolution kernels</param>
        /// <param name="kernels">The number of convolution kernels to be used</param>
        [Pure]
        internal Shape GetOutputShape(Shape input, (int X, int Y) size, int kernels)
        {
            int
                h = (input.H - size.X + 2 * VerticalPadding) / VerticalStride + 1,
                w = (input.W - size.Y + 2 * HorizontalPadding) / HorizontalStride + 1;

            Guard.IsTrue(h > 0 && w > 0, "The input convolution kernels can't be applied to the input tensor shape");

            return (kernels, h, w);
        }

        #region IEquatable<ConvolutionInfo>

        /// <inheritdoc/>
        public bool Equals(ConvolutionInfo other) => this == other;

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is ConvolutionInfo info && this == info;

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            Span<int> values = stackalloc int[] { (int)Mode, VerticalPadding, HorizontalPadding, VerticalStride, HorizontalStride };
            return values.GetContentHashCode();
        }

        /// <summary>
        /// Checks whether or not two <see cref="ConvolutionInfo"/> instances have the same parameters
        /// </summary>
        /// <param name="a">The first instance</param>
        /// <param name="b">The second instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator ==(in ConvolutionInfo a, in ConvolutionInfo b) => a.Mode == b.Mode &&
                                                                                      a.VerticalPadding == b.VerticalPadding && a.HorizontalPadding == b.HorizontalPadding &&
                                                                                      a.VerticalStride == b.VerticalStride && a.HorizontalStride == b.HorizontalStride;

        /// <summary>
        /// Checks whether or not two <see cref="ConvolutionInfo"/> instances have different parameters
        /// </summary>
        /// <param name="a">The first instance</param>
        /// <param name="b">The second instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator !=(in ConvolutionInfo a, in ConvolutionInfo b) => !(a == b);

        #endregion
    }
}
