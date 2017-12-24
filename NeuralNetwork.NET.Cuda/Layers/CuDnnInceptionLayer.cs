using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Cuda.Services;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using System;
using System.Runtime.CompilerServices;

namespace NeuralNetworkNET.Cuda.Layers
{
    /// <summary>
    /// A simplified inception module, with 4 pipelines combining 1x1 convolution, 1x1 + 3x3, 1x1 + 5x5 and pooling + 1x1
    /// </summary>
    internal sealed class CuDnnInceptionLayer : WeightedLayerBase
    {
        #region Parameters

        /// <sinheritdoc/>
        public override LayerType LayerType { get; } = LayerType.Inception;

        private readonly InceptionInfo _OperationInfo;

        /// <summary>
        /// Gets the info on the inception parameters used by the layer
        /// </summary>    
        public ref readonly InceptionInfo OperationInfo
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _OperationInfo;
        }

        #endregion

        #region cuDNN fields

        // The NCHW tensor info for the layer inputs
        [NotNull]
        private readonly TensorDescriptor InputDescription = new TensorDescriptor();

        #region 1x1 convolution

        // The NCHW info for the 1x1 convolution weights
        [NotNull]
        private readonly FilterDescriptor _1x1FilterDescription = new FilterDescriptor();

         // The info on the 1x1 convolution bias (one value per output channel)
        [NotNull]
        private readonly TensorDescriptor _1x1BiasDescription = new TensorDescriptor();

        // The first 1x1 convolution info
        [NotNull]
        private readonly ConvolutionDescriptor _1x1ConvolutionDescription = new ConvolutionDescriptor();

        // The NCHW tensor info for the outputs of the first 1x1 convolution
        [NotNull]
        private readonly TensorDescriptor _1x1OutputDescription = new TensorDescriptor();

        #endregion

        #region 3x3 secondary convolution

        // The NCHW info for the 3x3 convolution weights
        [NotNull]
        private readonly FilterDescriptor _3x3FilterDescription = new FilterDescriptor();

         // The info on the 3x3 convolution bias (one value per output channel)
        [NotNull]
        private readonly TensorDescriptor _3x3BiasDescription = new TensorDescriptor();

        // The first 3x3 convolution info
        [NotNull]
        private readonly ConvolutionDescriptor _3x3ConvolutionDescription = new ConvolutionDescriptor();

        // The NCHW tensor info for the outputs of the 3x3 convolution
        [NotNull]
        private readonly TensorDescriptor _3x3OutputDescription = new TensorDescriptor();

        #endregion

        #region 5x5 secondary convolution

        // The NCHW info for the 5x5 convolution weights
        [NotNull]
        private readonly FilterDescriptor _5x5FilterDescription = new FilterDescriptor();

         // The info on the 5x5 convolution bias (one value per output channel)
        [NotNull]
        private readonly TensorDescriptor _5x5BiasDescription = new TensorDescriptor();

        // The first 5x5 convolution info
        [NotNull]
        private readonly ConvolutionDescriptor _5x5ConvolutionDescription = new ConvolutionDescriptor();

        // The NCHW tensor info for the outputs of the 5x5 convolution
        [NotNull]
        private readonly TensorDescriptor _5x5OutputDescription = new TensorDescriptor();

        #endregion

        #region Pooling pipeline

        // The descriptor for the pooling operation performed by the layer
        [NotNull]
        private readonly PoolingDescriptor PoolingDescription = new PoolingDescriptor();

        // The NCHW tensor info for the pooling outputs
        [NotNull]
        private readonly TensorDescriptor PoolingOutputDescription = new TensorDescriptor();

        // The NCHW info for the secondary 1x1 convolution weights
        [NotNull]
        private readonly FilterDescriptor Secondary1x1FilterDescription = new FilterDescriptor();

        // The info on the secondary 1x1 convolution bias (one value per output channel)
        [NotNull]
        private readonly TensorDescriptor Secondary1x1BiasDescription = new TensorDescriptor();

        // The first secondary 1x1 convolution info
        [NotNull]
        private readonly ConvolutionDescriptor Secondary1x1ConvolutionDescription = new ConvolutionDescriptor();

        // The info on the secondary 1x1 convolution outputs
        [NotNull]
        private readonly TensorDescriptor Secondary1x1OutputDescription = new TensorDescriptor();

        #endregion

        /// <summary>
        /// Gets the <see cref="Dnn"/> instance for the current layer
        /// </summary>
        [NotNull]
        private readonly Dnn DnnInstance = DnnService.Instance;

        // cuDNN fields setup
        private void SetupCuDnnInfo()
        {

        }

        #endregion

        protected CuDnnInceptionLayer(in TensorInfo input, in TensorInfo output, [NotNull] float[] w, [NotNull] float[] b, ActivationFunctionType activation) : base(input, output, w, b, activation)
        {
        }

        #region Implementation

        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            throw new NotImplementedException();
        }

        public override void Backpropagate(in Tensor delta_1, in Tensor z, ActivationFunction activationPrime)
        {
            throw new NotImplementedException();
        }

        public override void ComputeGradient(in Tensor a, in Tensor delta, out Tensor dJdw, out Tensor dJdb)
        {
            throw new NotImplementedException();
        }

        #endregion

        public override INetworkLayer Clone()
        {
            throw new NotImplementedException();
        }
    }
}
