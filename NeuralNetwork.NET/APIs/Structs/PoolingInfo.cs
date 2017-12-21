using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using Newtonsoft.Json;
using System;

namespace NeuralNetworkNET.APIs.Structs
{
    /// <summary>
    /// A <see cref="struct"/> containing all the info on a pooling operation
    /// </summary>
    [JsonObject(MemberSerialization.Fields)]
    public readonly struct PoolingInfo
    {
        /// <summary>
        /// Gets the current pooling mode for the layer
        /// </summary>
        public readonly PoolingMode Mode;

        /// <summary>
        /// Gets the height of each input local receptive field
        /// </summary>
        public readonly int WindowHeight;

        /// <summary>
        /// Gets the width of each input local receptive field
        /// </summary>
        public readonly int WindowWidth;

        /// <summary>
        /// Gets the optional vertical padding for the pooling operation
        /// </summary>
        public readonly int VerticalPadding;

        /// <summary>
        /// Gets the optional horizontal padding for the pooling operation
        /// </summary>
        public readonly int HorizontalPadding;

        /// <summary>
        /// Gets the vertical stride length while sliding the receptive window over the input
        /// </summary>
        public readonly int VerticalStride;

        /// <summary>
        /// Gets the horizontal stride length while sliding the receptive window over the input
        /// </summary>
        public readonly int HorizontalStride;

        // Internal constructor
        private PoolingInfo(
            PoolingMode mode, int windowHeight, int windowWidth, 
            int verticalPadding, int horizontalPadding,
            int verticalStride, int horizontalStride)
        {
            if (windowHeight <= 0) throw new ArgumentOutOfRangeException(nameof(windowHeight), "The window height must be at least equal to 1");
            if (windowWidth <= 0) throw new ArgumentOutOfRangeException(nameof(windowWidth), "The window width must be at least equal to 1");
            if (verticalPadding < 0) throw new ArgumentOutOfRangeException(nameof(verticalPadding), "The vertical padding must be greater than or equal to 0");
            if (horizontalPadding < 0) throw new ArgumentOutOfRangeException(nameof(horizontalPadding), "The horizontal padding must be greater than or equal to 0");
            if (verticalStride < 1) throw new ArgumentOutOfRangeException(nameof(verticalStride), "The vertical stride must be at least equal to 1");
            if (horizontalStride < 1) throw new ArgumentOutOfRangeException(nameof(horizontalStride), "The horizontal stride must be at least equal to 1");

            Mode = mode;
            WindowHeight = windowHeight;
            WindowWidth = windowWidth;
            VerticalPadding = verticalPadding;
            HorizontalPadding = horizontalPadding;
            VerticalStride = verticalStride;
            HorizontalStride = horizontalStride;
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
    }
}
