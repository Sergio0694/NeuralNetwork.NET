using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using Newtonsoft.Json;
using System;
using System.Runtime.CompilerServices;
using NeuralNetworkNET.APIs.Delegates;

namespace NeuralNetworkNET.APIs.Structs
{
    /// <summary>
    /// A <see langword="struct"/> containing all the info on a convolution operation
    /// </summary>
    [JsonObject(MemberSerialization.Fields)]
    public readonly struct ConvolutionInfo : IEquatable<ConvolutionInfo>
    {
        /// <summary>
        /// The current convolution mode for the layer
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

        #region Constructors

        // Internal constructor
        private ConvolutionInfo(
            ConvolutionMode mode,
            int verticalPadding, int horizontalPadding,
            int verticalStride, int horizontalStride)
        {
            VerticalPadding = verticalPadding >= 0 ? verticalPadding : throw new ArgumentOutOfRangeException(nameof(verticalPadding), "The vertical padding must be greater than or equal to 0");
            HorizontalPadding = horizontalPadding >= 0 ? horizontalPadding : throw new ArgumentOutOfRangeException(nameof(horizontalPadding), "The horizontal padding must be greater than or equal to 0");
            VerticalStride = verticalStride >= 1 ? verticalStride : throw new ArgumentOutOfRangeException(nameof(verticalStride), "The vertical stride must be at least equal to 1");
            HorizontalStride = horizontalStride >= 1 ? horizontalStride : throw new ArgumentOutOfRangeException(nameof(horizontalStride), "The horizontal stride must be at least equal to 1");
            Mode = mode;
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
        /// Creates a new <see cref="ConvolutionInfoFactory"/> instance that returns a <see cref="ConvolutionInfo"/> value
        /// with the appropriate padding to keep the input size the same after the specified convolution operation
        /// </summary>
        /// <param name="mode">The desired convolution mode to use</param>
        /// <param name="verticalStride">The convolution vertical stride size</param>
        /// <param name="horizontalStride">The convolution horizontal stride size</param>
        [PublicAPI]
        [Pure]
        public static ConvolutionInfoFactory Same(
            ConvolutionMode mode = ConvolutionMode.Convolution,
            int verticalStride = 1, int horizontalStride = 1)
        {
            return (input, kernels) =>
            {
                int
                    verticalPadding = (input.Height * verticalStride - input.Height + kernels.X - verticalStride - 1) / 2 + 1,
                    horizontalPadding = (input.Width * horizontalStride - input.Width + kernels.Y - horizontalStride - 1) / 2 + 1;
                return new ConvolutionInfo(mode, verticalPadding, horizontalPadding, verticalStride, horizontalStride);
            };
        }

        #endregion

        /// <summary>
        /// Calculates the output size after applying a convolution operation to the input tensor
        /// </summary>
        /// <param name="input">The info on the input tensor</param>
        /// <param name="field">The size of the convolution kernels</param>
        /// <param name="kernels">The number of convolution kernels to be used</param>
        [Pure]
        internal TensorInfo GetForwardOutputTensorInfo(in TensorInfo input, (int X, int Y) field, int kernels)
        {
            int
                h = (input.Height - field.X + 2 * VerticalPadding) / VerticalStride + 1,
                w = (input.Width - field.Y + 2 * HorizontalPadding) / HorizontalStride + 1;
            if (h <= 0 || w <= 0) throw new InvalidOperationException("The input convolution kernels can't be applied to the input tensor shape");
            return new TensorInfo(h, w, kernels);
        }

        #region Equality

        /// <inheritdoc/>
        public bool Equals(ConvolutionInfo other) => this == other;

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is ConvolutionInfo info && this == info;

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            int hash = 17;
            unchecked
            {
                hash = hash * 31 + (int)Mode;
                hash = hash * 31 + VerticalPadding;
                hash = hash * 31 + HorizontalPadding;
                hash = hash * 31 + VerticalStride;
                hash = hash * 31 + HorizontalStride;
            }
            return hash;
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
