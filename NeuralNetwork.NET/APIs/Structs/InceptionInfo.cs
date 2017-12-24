using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using Newtonsoft.Json;
using System;
using System.Runtime.CompilerServices;

namespace NeuralNetworkNET.APIs.Structs
{
    /// <summary>
    /// A <see cref="struct"/> containing all the info on an inception module
    /// </summary>
    [JsonObject(MemberSerialization.Fields)]
    public readonly struct InceptionInfo : IEquatable<InceptionInfo>
    {
        #region Fields

        /// <summary>
        /// The number of 1x1 convolution kernels used in the first step of the forward pass
        /// </summary>
        public readonly int Primary1x1ConvolutionKernels;

        /// <summary>
        /// The number of 3x3 convolution kernels
        /// </summary>
        public readonly int Secondary3x3ConvolutionKernels;

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
        public readonly int Chained1x1AfterPoolingConvolutionKernels;

        #endregion

        #region Constructors

        // Internal constructor
        private InceptionInfo(int _1x1Kernels, int _3x3Kernels, int _5x5Kernels, PoolingMode poolingMode, int _1x1SecondaryKernels)
        {
            Primary1x1ConvolutionKernels = _1x1Kernels >= 1 ? _1x1Kernels : throw new ArgumentOutOfRangeException(nameof(_1x1Kernels), "The number of 1x1 kernels must be at least 1");
            Secondary3x3ConvolutionKernels = _3x3Kernels >= 1 ? _3x3Kernels : throw new ArgumentOutOfRangeException(nameof(_3x3Kernels), "The number of 3x3 kernels must be at least 1");
            Secondary5x5ConvolutionKernels = _5x5Kernels >= 1 ? _5x5Kernels : throw new ArgumentOutOfRangeException(nameof(_5x5Kernels), "The number of 5x5 kernels must be at least 1");
            Chained1x1AfterPoolingConvolutionKernels = _1x1SecondaryKernels >= 1 ? _1x1SecondaryKernels : throw new ArgumentOutOfRangeException(nameof(_1x1SecondaryKernels), "The number of secondary 1x1 kernels must be at least 1");
            Pooling = poolingMode;
        }

        /// <summary>
        /// Creates a new inception layer description with the input parameters
        /// </summary>
        /// <param name="_1x1Kernels">The number of 1x1 primary convolution kernels</param>
        /// <param name="_3x3Kernels">The number of 3x3 convolution kernels</param>
        /// <param name="_5x5Kernels">The number of 5x5 convolution kernels</param>
        /// <param name="poolingMode">The pooling mode for the pooling pipeline</param>
        /// <param name="_1x1SecondaryKernels">The number of secondary 1x1 convolution kernels</param>
        [PublicAPI]
        [Pure]
        public static InceptionInfo New(
            int _1x1Kernels, int _3x3Kernels, int _5x5Kernels, 
            PoolingMode poolingMode, int _1x1SecondaryKernels)
            => new InceptionInfo(_1x1Kernels, _3x3Kernels, _5x5Kernels, poolingMode, _1x1SecondaryKernels);

        #endregion

        #region Equality

        /// <inheritdoc/>
        public bool Equals(InceptionInfo other) => this == other;

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is InceptionInfo info ? this == info : false;

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            int hash = 17;
            unchecked
            {
                hash = hash * 31 + Primary1x1ConvolutionKernels;
                hash = hash * 31 + Chained1x1AfterPoolingConvolutionKernels;
                hash = hash * 31 + Secondary3x3ConvolutionKernels;
                hash = hash * 31 + Secondary5x5ConvolutionKernels;
                hash = hash * 31 + (int)Pooling;
            }
            return hash;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator ==(in InceptionInfo a, in InceptionInfo b) => a.Primary1x1ConvolutionKernels == b.Primary1x1ConvolutionKernels &&
                                                                                  a.Chained1x1AfterPoolingConvolutionKernels == b.Chained1x1AfterPoolingConvolutionKernels && 
                                                                                  a.Secondary3x3ConvolutionKernels == b.Secondary3x3ConvolutionKernels &&
                                                                                  a.Secondary5x5ConvolutionKernels == b.Secondary5x5ConvolutionKernels && 
                                                                                  a.Pooling == b.Pooling;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator !=(in InceptionInfo a, in InceptionInfo b) => !(a == b);

        #endregion
    }
}
