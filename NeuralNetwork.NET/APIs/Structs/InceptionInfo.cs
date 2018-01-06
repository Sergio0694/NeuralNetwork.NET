using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using Newtonsoft.Json;
using System;
using System.Runtime.CompilerServices;

namespace NeuralNetworkNET.APIs.Structs
{
    /// <summary>
    /// A <see langword="struct"/> containing all the info on an inception module
    /// </summary>
    [JsonObject(MemberSerialization.Fields)]
    public readonly struct InceptionInfo : IEquatable<InceptionInfo>
    {
        #region Fields and properties

        /// <summary>
        /// The number of 1x1 convolution kernels used in the first step of the forward pass
        /// </summary>
        public readonly int Primary1x1ConvolutionKernels;

        /// <summary>
        /// The number of 1x1 convolution kernels before the 3x3 convolution
        /// </summary>
        public readonly int Primary3x3Reduce1x1ConvolutionKernels;

        /// <summary>
        /// The number of 3x3 convolution kernels
        /// </summary>
        public readonly int Secondary3x3ConvolutionKernels;

        /// <summary>
        /// The number of 1x1 convolution kernels before the 5x5 convolution
        /// </summary>
        public readonly int Primary5x5Reduce1x1ConvolutionKernels;

        /// <summary>
        /// The number of 5x5 convolution kernels
        /// </summary>
        public readonly int Secondary5x5ConvolutionKernels;

        /// <summary>
        /// The kind of pooling operation performed on the layer
        /// </summary>
        public readonly PoolingMode Pooling;

        /// <summary>
        /// The number of 1x1 convolution kernels after the pooling operation
        /// </summary>
        public readonly int Secondary1x1AfterPoolingConvolutionKernels;

        /// <summary>
        /// Gets the number of output channels after the depth concatenation
        /// </summary>
        public int OutputChannels
        {
            [Pure]
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Primary1x1ConvolutionKernels + Secondary3x3ConvolutionKernels + Secondary5x5ConvolutionKernels + Secondary1x1AfterPoolingConvolutionKernels;
        }

        /// <summary>
        /// Gets the total number of convolution kernels for the current instance
        /// </summary>
        public int ConvolutionKernels
        {
            [Pure]
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Primary1x1ConvolutionKernels + Primary3x3Reduce1x1ConvolutionKernels + Secondary3x3ConvolutionKernels + Primary5x5Reduce1x1ConvolutionKernels + Secondary5x5ConvolutionKernels + Secondary1x1AfterPoolingConvolutionKernels;
        }

        #endregion

        #region Constructors

        // Internal constructor
        private InceptionInfo(int _1x1Kernels, int _3x3Reduce1x1Kernels, int _3x3Kernels, int _5x5Reduce1x1Kernels, int _5x5Kernels, PoolingMode poolingMode, int _1x1SecondaryKernels)
        {
            Primary1x1ConvolutionKernels = _1x1Kernels >= 1 ? _1x1Kernels : throw new ArgumentOutOfRangeException(nameof(_1x1Kernels), "The number of 1x1 kernels must be at least 1");
            Primary3x3Reduce1x1ConvolutionKernels = _3x3Reduce1x1Kernels >= 1 ? _3x3Reduce1x1Kernels : throw new ArgumentOutOfRangeException(nameof(_3x3Reduce1x1Kernels), "The number of 3x3 reduction 1x1 kernels must be at least 1");
            Secondary3x3ConvolutionKernels = _3x3Kernels >= 1 ? _3x3Kernels : throw new ArgumentOutOfRangeException(nameof(_3x3Kernels), "The number of 3x3 kernels must be at least 1");
            Primary5x5Reduce1x1ConvolutionKernels = _5x5Reduce1x1Kernels >= 1 ? _5x5Reduce1x1Kernels : throw new ArgumentOutOfRangeException(nameof(_3x3Kernels), "The number of 5x5 reduction 1x1 kernels must be at least 1");
            Secondary5x5ConvolutionKernels = _5x5Kernels >= 1 ? _5x5Kernels : throw new ArgumentOutOfRangeException(nameof(_5x5Kernels), "The number of 5x5 kernels must be at least 1");
            Secondary1x1AfterPoolingConvolutionKernels = _1x1SecondaryKernels >= 1 ? _1x1SecondaryKernels : throw new ArgumentOutOfRangeException(nameof(_1x1SecondaryKernels), "The number of secondary 1x1 kernels must be at least 1");
            Pooling = poolingMode;
        }

        /// <summary>
        /// Creates a new inception layer description with the input parameters
        /// </summary>
        /// <param name="_1x1Kernels">The number of 1x1 primary convolution kernels</param>
        /// <param name="_3x3Reduce1x1Kernels">The number of 3x3 reduction 1x1 kernels</param>
        /// <param name="_3x3Kernels">The number of 3x3 convolution kernels</param>
        /// <param name="_5x5Reduce1x1Kernels">The number of 5x5 reduction 1x1 kernels</param>
        /// <param name="_5x5Kernels">The number of 5x5 convolution kernels</param>
        /// <param name="poolingMode">The pooling mode for the pooling pipeline</param>
        /// <param name="_1x1SecondaryKernels">The number of secondary 1x1 convolution kernels</param>
        [PublicAPI]
        [Pure]
        public static InceptionInfo New(
            int _1x1Kernels, int _3x3Reduce1x1Kernels, int _3x3Kernels, int _5x5Reduce1x1Kernels, int _5x5Kernels, 
            PoolingMode poolingMode, int _1x1SecondaryKernels)
            => new InceptionInfo(_1x1Kernels, _3x3Reduce1x1Kernels, _3x3Kernels, _5x5Reduce1x1Kernels, _5x5Kernels, poolingMode, _1x1SecondaryKernels);

        #endregion

        #region Equality

        /// <inheritdoc/>
        public bool Equals(InceptionInfo other) => this == other;

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is InceptionInfo info && this == info;

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            int hash = 17;
            unchecked
            {
                hash = hash * 31 + Primary1x1ConvolutionKernels;
                hash = hash * 31 + Primary3x3Reduce1x1ConvolutionKernels;
                hash = hash * 31 + Secondary3x3ConvolutionKernels;
                hash = hash * 31 + Primary5x5Reduce1x1ConvolutionKernels;
                hash = hash * 31 + Secondary5x5ConvolutionKernels;
                hash = hash * 31 + Secondary1x1AfterPoolingConvolutionKernels;
                hash = hash * 31 + (int)Pooling;
            }
            return hash;
        }

        /// <summary>
        /// Checks whether or not two <see cref="InceptionInfo"/> instances have the same parameters
        /// </summary>
        /// <param name="a">The first instance</param>
        /// <param name="b">The second instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator ==(in InceptionInfo a, in InceptionInfo b) => a.Primary1x1ConvolutionKernels == b.Primary1x1ConvolutionKernels &&
                                                                                  a.Primary3x3Reduce1x1ConvolutionKernels == b.Primary3x3Reduce1x1ConvolutionKernels && 
                                                                                  a.Secondary3x3ConvolutionKernels == b.Secondary3x3ConvolutionKernels &&
                                                                                  a.Primary5x5Reduce1x1ConvolutionKernels == b.Primary5x5Reduce1x1ConvolutionKernels &&
                                                                                  a.Secondary5x5ConvolutionKernels == b.Secondary5x5ConvolutionKernels && 
                                                                                  a.Secondary1x1AfterPoolingConvolutionKernels == b.Secondary1x1AfterPoolingConvolutionKernels &&
                                                                                  a.Pooling == b.Pooling;

        /// <summary>
        /// Checks whether or not two <see cref="InceptionInfo"/> instances have different parameters
        /// </summary>
        /// <param name="a">The first instance</param>
        /// <param name="b">The second instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator !=(in InceptionInfo a, in InceptionInfo b) => !(a == b);

        #endregion
    }
}
