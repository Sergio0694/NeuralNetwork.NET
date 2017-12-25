using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Cuda.Extensions;
using NeuralNetworkNET.Cuda.Services;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Networks.Implementations.Layers.Helpers;
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

        #region 3x3 reduce 1x1 convolution

        // The NCHW info for the 3x3 reduce 1x1 convolution weights
        [NotNull]
        private readonly FilterDescriptor _3x3Reduce1x1FilterDescription = new FilterDescriptor();

         // The info on the 3x3 reduce 1x1 convolution bias (one value per output channel)
        [NotNull]
        private readonly TensorDescriptor _3x3Reduce1x1BiasDescription = new TensorDescriptor();

        // The NCHW tensor info for the outputs of the 3x3 reduce 1x1 convolution
        [NotNull]
        private readonly TensorDescriptor _3x3Reduce1x1OutputDescription = new TensorDescriptor();

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

        #region 3x3 reduce 1x1 convolution

        // The NCHW info for the 5x5 reduce 1x1 convolution weights
        [NotNull]
        private readonly FilterDescriptor _5x5Reduce1x1FilterDescription = new FilterDescriptor();

         // The info on the 5x5 reduce 1x1 convolution bias (one value per output channel)
        [NotNull]
        private readonly TensorDescriptor _5x5Reduce1x1BiasDescription = new TensorDescriptor();

        // The NCHW tensor info for the outputs of the 5x5 reduce 1x1 convolution
        [NotNull]
        private readonly TensorDescriptor _5x5Reduce1x1OutputDescription = new TensorDescriptor();

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

        // The info on the secondary 1x1 convolution outputs
        [NotNull]
        private readonly TensorDescriptor Secondary1x1OutputDescription = new TensorDescriptor();

        #endregion

        // The shared ReLU activation description for the current layer
        [NotNull]
        private readonly ActivationDescriptor ActivationDescription = new ActivationDescriptor();

        /// <summary>
        /// Gets the <see cref="Dnn"/> instance for the current layer
        /// </summary>
        [NotNull]
        private readonly Dnn DnnInstance = DnnService.Instance;

        // cuDNN fields setup
        private void SetupCuDnnInfo()
        {
            // First 1x1 convolution
            _1x1FilterDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, _OperationInfo.Primary1x1ConvolutionKernels, InputInfo.Channels, 1, 1);
            _1x1BiasDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, 1, _OperationInfo.Primary1x1ConvolutionKernels, 1, 1);

            // 3x3 reduce 1x1 convolution
            _3x3Reduce1x1FilterDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, _OperationInfo.Primary3x3Reduce1x1ConvolutionKernels, InputInfo.Channels, 1, 1);
            _3x3Reduce1x1BiasDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, 1, _OperationInfo.Primary3x3Reduce1x1ConvolutionKernels, 1, 1);

            // 3x3 convolution
            _3x3ConvolutionDescription.Set2D(1, 1, 1, 1, 1, 1, Alea.cuDNN.ConvolutionMode.CROSS_CORRELATION); // 1-padding to keep size
            _3x3FilterDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, _OperationInfo.Secondary3x3ConvolutionKernels, _OperationInfo.Primary3x3Reduce1x1ConvolutionKernels, 3, 3);
            _3x3BiasDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, 1, _OperationInfo.Secondary3x3ConvolutionKernels, 1, 1);

            // 5x5 reduce 1x1 convolution
            _5x5Reduce1x1FilterDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, _OperationInfo.Primary5x5Reduce1x1ConvolutionKernels, InputInfo.Channels, 1, 1);
            _5x5Reduce1x1BiasDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, 1, _OperationInfo.Primary5x5Reduce1x1ConvolutionKernels, 1, 1);

            // 5x5 convolution
            _5x5ConvolutionDescription.Set2D(2, 2, 1, 1, 1, 1, Alea.cuDNN.ConvolutionMode.CROSS_CORRELATION);
            _5x5FilterDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, _OperationInfo.Secondary5x5ConvolutionKernels, _OperationInfo.Primary5x5Reduce1x1ConvolutionKernels, 5, 5);
            _5x5BiasDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, 1, _OperationInfo.Secondary5x5ConvolutionKernels, 1, 1);

            // Pooling
            PoolingDescription.Set2D(Alea.cuDNN.PoolingMode.AVERAGE_COUNT_EXCLUDE_PADDING, NanPropagation.PROPAGATE_NAN, 3, 3, 1, 1, 1, 1);
            
            // Secondary 1x1 convolution
            Secondary1x1FilterDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, InputInfo.Channels, _OperationInfo.Secondary1x1AfterPoolingConvolutionKernels, 1, 1);
            Secondary1x1BiasDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, 1, _OperationInfo.Secondary1x1AfterPoolingConvolutionKernels, 1, 1);

            // Activation
            ActivationDescription.Set(ActivationMode.RELU, NanPropagation.PROPAGATE_NAN, 0);
        }

        #endregion

        internal CuDnnInceptionLayer(in TensorInfo input, in InceptionInfo info, BiasInitializationMode biasMode = BiasInitializationMode.Zero)
            : base(input, new TensorInfo(input.Height, input.Width, info.OutputChannels),
                  WeightsProvider.NewInceptionWeights(input, info),
                  WeightsProvider.NewBiases(info.ConvolutionKernels, biasMode),
                  ActivationFunctionType.ReLU)
        {
            _OperationInfo = info;
            SetupCuDnnInfo();
        }

        internal CuDnnInceptionLayer(in TensorInfo input, in InceptionInfo info, [NotNull] float[] w, [NotNull] float[] b) 
            : base(input, new TensorInfo(input.Height, input.Width, info.OutputChannels), w, b, ActivationFunctionType.ReLU)
        {
            _OperationInfo = info;
            SetupCuDnnInfo();
        }

        #region Implementation

        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            Tensor.New(x.Entities, OutputInfo.Size, out z);
            Tensor.New(x.Entities, OutputInfo.Size, out a);
            using (DeviceMemory<float>
                    w_gpu = DnnInstance.Gpu.AllocateDevice(Weights),
                    b_gpu = DnnInstance.Gpu.AllocateDevice(Biases))
            {
                using (DeviceMemory<float> _1x1Output_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels))
                {
                    // First 1x1 convolution
                    using (DeviceMemory<float> x_gpu = DnnInstance.Gpu.AllocateDevice(x))
                    {
                        // Descriptors setup and first 1x1 convolution
                        InputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, InputInfo.Channels, InputInfo.Height, InputInfo.Width);
                        _1x1OutputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, OperationInfo.Primary1x1ConvolutionKernels, InputInfo.Height, InputInfo.Width);
                        DnnInstance.GetConvolutionForwardAlgorithm(InputDescription, _1x1FilterDescription, _1x1ConvolutionDescription, _1x1OutputDescription, ConvolutionFwdPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionFwdAlgo algorithm);
                        DnnInstance.GetConvolutionForwardWorkspaceSize(InputDescription, _1x1FilterDescription, _1x1ConvolutionDescription, _1x1OutputDescription, algorithm, out IntPtr size);
                        using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                        {
                            DnnInstance.ConvolutionForward(1, InputDescription, x_gpu.Ptr, _1x1FilterDescription, w_gpu.Ptr, _1x1ConvolutionDescription, algorithm, workspace_gpu.Ptr, size, 0, _1x1OutputDescription, _1x1Output_gpu.Ptr);                            
                        }
                        DnnInstance.AddTensor(1, _1x1BiasDescription, b_gpu.Ptr, 1, _1x1OutputDescription, _1x1Output_gpu.Ptr);
                        _1x1Output_gpu.CopyToRows(z, 0, InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels);

                        // 1x1 convolution activation
                        DnnInstance.ActivationForward(ActivationDescription, 1, _1x1OutputDescription, _1x1Output_gpu.Ptr, 0, _1x1OutputDescription, _1x1Output_gpu.Ptr);
                        _1x1Output_gpu.CopyToRows(a, 0, InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels);
                    }

                    // 3x3 convolution
                    using (DeviceMemory<float> _3x3Output_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * InputInfo.SliceSize * OperationInfo.Secondary3x3ConvolutionKernels))
                    {
                        _3x3OutputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, OperationInfo.Secondary3x3ConvolutionKernels, InputInfo.Height, InputInfo.Width);
                        DnnInstance.GetConvolutionForwardAlgorithm(_1x1OutputDescription, _3x3FilterDescription, _3x3ConvolutionDescription, _3x3OutputDescription, ConvolutionFwdPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionFwdAlgo algorithm);
                        DnnInstance.GetConvolutionForwardWorkspaceSize(_1x1OutputDescription, _3x3FilterDescription, _3x3ConvolutionDescription, _3x3OutputDescription, algorithm, out IntPtr size);
                        using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                        {
                            DnnInstance.ConvolutionForward(1, _1x1OutputDescription, _1x1Output_gpu.Ptr, _3x3FilterDescription, w_gpu.Ptr + InputInfo.Channels * OperationInfo.Primary1x1ConvolutionKernels, _3x3ConvolutionDescription, algorithm, workspace_gpu.Ptr, size, 0, _3x3OutputDescription, _3x3Output_gpu.Ptr);                            
                        }
                        DnnInstance.AddTensor(1, _3x3BiasDescription, b_gpu.Ptr + OperationInfo.Primary1x1ConvolutionKernels, 1, _3x3OutputDescription, _3x3Output_gpu.Ptr);
                        _3x3Output_gpu.CopyToRows(z, InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels, InputInfo.SliceSize * OperationInfo.Secondary3x3ConvolutionKernels);

                        // 3x3 convolution activation
                        DnnInstance.ActivationForward(ActivationDescription, 1, _3x3OutputDescription, _3x3Output_gpu.Ptr, 0, _3x3OutputDescription, _3x3Output_gpu.Ptr);
                        _3x3Output_gpu.CopyToRows(a, InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels, InputInfo.SliceSize * OperationInfo.Secondary3x3ConvolutionKernels);
                    }

                    // 5x5 convolution
                    using (DeviceMemory<float> _5x5Output_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * InputInfo.SliceSize * OperationInfo.Secondary5x5ConvolutionKernels))
                    {
                        _5x5OutputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, OperationInfo.Secondary5x5ConvolutionKernels, InputInfo.Height, InputInfo.Width);
                        DnnInstance.GetConvolutionForwardAlgorithm(_1x1OutputDescription, _5x5FilterDescription, _5x5ConvolutionDescription, _5x5OutputDescription, ConvolutionFwdPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionFwdAlgo algorithm);
                        DnnInstance.GetConvolutionForwardWorkspaceSize(_1x1OutputDescription, _5x5FilterDescription, _5x5ConvolutionDescription, _5x5OutputDescription, algorithm, out IntPtr size);
                        using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                        {
                            DnnInstance.ConvolutionForward(1, _1x1OutputDescription, _1x1Output_gpu.Ptr, _5x5FilterDescription, w_gpu.Ptr + InputInfo.Channels * OperationInfo.Primary1x1ConvolutionKernels + 3 * 3 * OperationInfo.Primary1x1ConvolutionKernels * OperationInfo.Secondary3x3ConvolutionKernels, _5x5ConvolutionDescription, algorithm, workspace_gpu.Ptr, size, 0, _5x5OutputDescription, _5x5Output_gpu.Ptr);                            
                        }
                        DnnInstance.AddTensor(1, _5x5BiasDescription, b_gpu.Ptr + OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels, 1, _5x5OutputDescription, _5x5Output_gpu.Ptr);
                        _5x5Output_gpu.CopyToRows(z, InputInfo.SliceSize * (OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels), InputInfo.SliceSize * OperationInfo.Secondary5x5ConvolutionKernels);

                        // 3x3 convolution activation
                        DnnInstance.ActivationForward(ActivationDescription, 1, _5x5OutputDescription, _5x5Output_gpu.Ptr, 0, _5x5OutputDescription, _5x5Output_gpu.Ptr);
                        _5x5Output_gpu.CopyToRows(a, InputInfo.SliceSize * (OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels), InputInfo.SliceSize * OperationInfo.Secondary5x5ConvolutionKernels);
                    }
                }
                
                // Pooling pipeline
                PoolingOutputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, InputInfo.Channels, InputInfo.Height, InputInfo.Width);
                using (DeviceMemory<float> y_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Size))
                {
                    // Pooling
                    using (DeviceMemory<float> x_gpu = DnnInstance.Gpu.AllocateDevice(x))
                    {
                        DnnInstance.PoolingForward(PoolingDescription, 1, InputDescription, x_gpu.Ptr, 0, InputDescription, y_gpu.Ptr);
                    }

                    // 1x1 convolution
                    using (DeviceMemory<float> _1x1Output_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * InputInfo.SliceSize)) // TODO
                    {
                        _1x1OutputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, -1, InputInfo.Height, InputInfo.Width);
                        DnnInstance.GetConvolutionForwardAlgorithm(InputDescription, Secondary1x1FilterDescription, _1x1ConvolutionDescription, Secondary1x1OutputDescription, ConvolutionFwdPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionFwdAlgo algorithm);
                        DnnInstance.GetConvolutionForwardWorkspaceSize(InputDescription, Secondary1x1FilterDescription, _1x1ConvolutionDescription, Secondary1x1OutputDescription, algorithm, out IntPtr size);
                        using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                        {
                            DnnInstance.ConvolutionForward(1, InputDescription, y_gpu.Ptr, Secondary1x1FilterDescription, w_gpu.Ptr + InputInfo.Channels * OperationInfo.Primary1x1ConvolutionKernels + 3 * 3 * OperationInfo.Primary1x1ConvolutionKernels * OperationInfo.Secondary3x3ConvolutionKernels + 5 * 5 * OperationInfo.Primary1x1ConvolutionKernels * OperationInfo.Secondary5x5ConvolutionKernels, _1x1ConvolutionDescription, algorithm, workspace_gpu.Ptr, size, 0, Secondary1x1OutputDescription, _1x1Output_gpu.Ptr);                            
                        }
                        DnnInstance.AddTensor(1, Secondary1x1BiasDescription, b_gpu.Ptr + OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels + OperationInfo.Secondary5x5ConvolutionKernels, 1, Secondary1x1OutputDescription, _1x1Output_gpu.Ptr);
                        _1x1Output_gpu.CopyToRows(z, InputInfo.SliceSize * (OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels + OperationInfo.Secondary5x5ConvolutionKernels), InputInfo.SliceSize); // TODO

                        // 1x1 convolution activation
                        DnnInstance.ActivationForward(ActivationDescription, 1, Secondary1x1OutputDescription, _1x1Output_gpu.Ptr, 0, Secondary1x1OutputDescription, _1x1Output_gpu.Ptr);
                        _1x1Output_gpu.CopyToRows(a, InputInfo.SliceSize * (OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels + OperationInfo.Secondary5x5ConvolutionKernels), InputInfo.SliceSize); // TODO
                    }
                }
            }
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

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new CuDnnInceptionLayer(InputInfo, OperationInfo, Weights, Biases);
    }
}
