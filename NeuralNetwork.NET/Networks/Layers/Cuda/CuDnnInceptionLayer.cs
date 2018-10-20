using System;
using System.Runtime.CompilerServices;
using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Layers.Abstract;
using NeuralNetworkNET.Networks.Layers.Initialization;

namespace NeuralNetworkNET.Networks.Layers.Cuda
{
    /// <summary>
    /// An inception module with 4 pipelines combining 1x1 convolution, 1x1 + 3x3, 1x1 + 5x5 and pooling + 1x1, see <a href="https://arxiv.org/pdf/1409.4842.pdf">arxiv.org/pdf/1409.4842.pdf</a>
    /// </summary>
    internal sealed class CuDnnInceptionLayer : WeightedLayerBase, IDisposable
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

        #region Private fields and parameters

        // 1x1 convolution weights on first pipeline
        private int _1x1Weights
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => InputInfo.Channels * OperationInfo.Primary1x1ConvolutionKernels;
        }

        // 1x1 convolution weights on 3x3 pipeline
        private int _3x3Reduce1x1Weights
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => InputInfo.Channels * OperationInfo.Primary3x3Reduce1x1ConvolutionKernels;
        }

        // 3x3 convolution weights
        private int _3x3Weights
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => 3 * 3 * OperationInfo.Primary3x3Reduce1x1ConvolutionKernels * OperationInfo.Secondary3x3ConvolutionKernels;
        }

        // 1x1 convolution weights on 5x5 pipeline
        private int _5x5Reduce1x1Weights
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => InputInfo.Channels * OperationInfo.Primary5x5Reduce1x1ConvolutionKernels;
        }

        // 5x5 convolution weights
        private int _5x5Weights
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => 5 * 5 * OperationInfo.Primary5x5Reduce1x1ConvolutionKernels * OperationInfo.Secondary5x5ConvolutionKernels;
        }

        // 1x1 convolution weights on pooling pipeline
        private int Secondary1x1Weights
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => InputInfo.Channels * OperationInfo.Secondary1x1AfterPoolingConvolutionKernels;
        }

        // 3x3 reduction 1x1 convolution activity
        private Tensor _3x3Reduce1x1Z;

        // 3x3 reduction 1x1 convolution activation
        private Tensor _3x3Reduce1x1A;

        // 5x5 reduction 1x1 convolution activity
        private Tensor _5x5Reduce1x1Z;

        // 5x5 reduction 1x1 convolution activation
        private Tensor _5x5Reduce1x1A;

        // Pooling output activity
        private Tensor _PoolingZ;

        // Pooling output activation
        private Tensor _PoolingA;

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

        /// <summary>
        /// Gets the <see cref="Dnn"/> instance for the current layer
        /// </summary>
        [NotNull]
        private readonly Dnn DnnInstance = CuDnnService.Instance;

        // cuDNN fields setup
        private void SetupCuDnnInfo()
        {
            // First 1x1 convolution
            _1x1ConvolutionDescription.Set2D(0, 0, 1, 1, 1, 1, Alea.cuDNN.ConvolutionMode.CROSS_CORRELATION);
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
            PoolingDescription.Set2D((Alea.cuDNN.PoolingMode)OperationInfo.Pooling, NanPropagation.PROPAGATE_NAN, 3, 3, 1, 1, 1, 1);

            // Secondary 1x1 convolution
            Secondary1x1FilterDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, _OperationInfo.Secondary1x1AfterPoolingConvolutionKernels, InputInfo.Channels, 1, 1);
            Secondary1x1BiasDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, 1, _OperationInfo.Secondary1x1AfterPoolingConvolutionKernels, 1, 1);
        }

        #endregion

        internal CuDnnInceptionLayer(in TensorInfo input, in InceptionInfo info, BiasInitializationMode biasMode = BiasInitializationMode.Zero)
            : base(input, new TensorInfo(input.Height, input.Width, info.OutputChannels),
                  WeightsProvider.NewInceptionWeights(input, info),
                  WeightsProvider.NewBiases(info.ConvolutionKernels, biasMode),
                  ActivationType.ReLU)
        {
            _OperationInfo = info;
            SetupCuDnnInfo();
        }

        internal CuDnnInceptionLayer(in TensorInfo input, in InceptionInfo info, [NotNull] float[] w, [NotNull] float[] b) 
            : base(input, new TensorInfo(input.Height, input.Width, info.OutputChannels), w, b, ActivationType.ReLU)
        {
            _OperationInfo = info;
            SetupCuDnnInfo();
        }

        #region Implementation

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            Tensor.New(x.Entities, OutputInfo.Size, out z);
            Tensor.New(x.Entities, OutputInfo.Size, out a);
            using (DeviceMemory<float>
                    w_gpu = DnnInstance.Gpu.AllocateDevice(Weights),
                    b_gpu = DnnInstance.Gpu.AllocateDevice(Biases))
            {
                // Pointers
                deviceptr<float> pw_gpu = w_gpu.Ptr, pb_gpu = b_gpu.Ptr;

                #region First 1x1 convolution

                using (DeviceMemory<float> y_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels))
                {
                    // Descriptors setup and first 1x1 convolution
                    InputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, InputInfo.Channels, InputInfo.Height, InputInfo.Width);
                    _1x1OutputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, OperationInfo.Primary1x1ConvolutionKernels, InputInfo.Height, InputInfo.Width);
                    DnnInstance.GetConvolutionForwardAlgorithm(InputDescription, _1x1FilterDescription, _1x1ConvolutionDescription, _1x1OutputDescription, ConvolutionFwdPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionFwdAlgo algorithm);
                    DnnInstance.GetConvolutionForwardWorkspaceSize(InputDescription, _1x1FilterDescription, _1x1ConvolutionDescription, _1x1OutputDescription, algorithm, out IntPtr size);
                    using (DeviceMemory<float> x_gpu = DnnInstance.Gpu.AllocateDevice(x))
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionForward(1, InputDescription, x_gpu.Ptr, _1x1FilterDescription, pw_gpu, _1x1ConvolutionDescription, algorithm, workspace_gpu.Ptr, size, 0, _1x1OutputDescription, y_gpu.Ptr);                            
                    }
                    DnnInstance.AddTensor(1, _1x1BiasDescription, pb_gpu, 1, _1x1OutputDescription, y_gpu.Ptr);
                    y_gpu.CopyTo(z, 0, InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels);

                    // 1x1 convolution activation
                    DnnInstance.ActivationForward(x.Entities, InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels, y_gpu.Ptr, y_gpu.Ptr, ActivationFunctions.Activation);
                    y_gpu.CopyTo(a, 0, InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels);
                }

                #endregion

                #region 1x1 + 3x3 convolution

                using (DeviceMemory<float> 
                    y1x1_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * InputInfo.SliceSize * OperationInfo.Primary3x3Reduce1x1ConvolutionKernels),
                    y_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * InputInfo.SliceSize * OperationInfo.Secondary3x3ConvolutionKernels))
                {
                    // 1x1 convolution
                    _3x3Reduce1x1OutputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, OperationInfo.Primary3x3Reduce1x1ConvolutionKernels, InputInfo.Height, InputInfo.Width);
                    DnnInstance.GetConvolutionForwardAlgorithm(InputDescription, _3x3Reduce1x1FilterDescription, _1x1ConvolutionDescription, _3x3Reduce1x1OutputDescription, ConvolutionFwdPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionFwdAlgo algorithm);
                    DnnInstance.GetConvolutionForwardWorkspaceSize(InputDescription, _3x3Reduce1x1FilterDescription, _1x1ConvolutionDescription, _3x3Reduce1x1OutputDescription, algorithm, out IntPtr size);
                    using (DeviceMemory<float> x_gpu = DnnInstance.Gpu.AllocateDevice(x))
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionForward(1, InputDescription, x_gpu.Ptr, _3x3Reduce1x1FilterDescription, pw_gpu += _1x1Weights, _1x1ConvolutionDescription, algorithm, workspace_gpu.Ptr, size, 0, _3x3Reduce1x1OutputDescription, y1x1_gpu.Ptr);                            
                    }
                    DnnInstance.AddTensor(1, _3x3Reduce1x1BiasDescription, pb_gpu += OperationInfo.Primary1x1ConvolutionKernels, 1, _3x3Reduce1x1OutputDescription, y1x1_gpu.Ptr);
                    _3x3Reduce1x1Z.TryFree();
                    y1x1_gpu.CopyToHost(x.Entities, InputInfo.SliceSize * OperationInfo.Primary3x3Reduce1x1ConvolutionKernels, out _3x3Reduce1x1Z);
                    DnnInstance.ActivationForward(x.Entities, InputInfo.SliceSize * OperationInfo.Primary3x3Reduce1x1ConvolutionKernels, y1x1_gpu.Ptr, y1x1_gpu.Ptr, ActivationFunctions.Activation);
                    _3x3Reduce1x1A.TryFree();
                    y1x1_gpu.CopyToHost(x.Entities, InputInfo.SliceSize * OperationInfo.Primary3x3Reduce1x1ConvolutionKernels, out _3x3Reduce1x1A);

                    // 3x3 convolution
                    _3x3OutputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, OperationInfo.Secondary3x3ConvolutionKernels, InputInfo.Height, InputInfo.Width);
                    DnnInstance.GetConvolutionForwardAlgorithm(_3x3Reduce1x1OutputDescription, _3x3FilterDescription, _3x3ConvolutionDescription, _3x3OutputDescription, ConvolutionFwdPreference.PREFER_FASTEST, IntPtr.Zero, out algorithm);
                    DnnInstance.GetConvolutionForwardWorkspaceSize(_3x3Reduce1x1OutputDescription, _3x3FilterDescription, _3x3ConvolutionDescription, _3x3OutputDescription, algorithm, out size);
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionForward(1, _3x3Reduce1x1OutputDescription, y1x1_gpu.Ptr, _3x3FilterDescription, pw_gpu += _3x3Reduce1x1Weights, _3x3ConvolutionDescription, algorithm, workspace_gpu.Ptr, size, 0, _3x3OutputDescription, y_gpu.Ptr);      
                    }
                    DnnInstance.AddTensor(1, _3x3BiasDescription, pb_gpu += OperationInfo.Primary3x3Reduce1x1ConvolutionKernels, 1, _3x3OutputDescription, y_gpu.Ptr);
                    y_gpu.CopyTo(z, InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels, InputInfo.SliceSize * OperationInfo.Secondary3x3ConvolutionKernels);

                    // Activation
                    DnnInstance.ActivationForward(x.Entities, InputInfo.SliceSize * OperationInfo.Secondary3x3ConvolutionKernels, y_gpu.Ptr, y_gpu.Ptr, ActivationFunctions.Activation);
                    y_gpu.CopyTo(a, InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels, InputInfo.SliceSize * OperationInfo.Secondary3x3ConvolutionKernels);
                }

                #endregion

                #region 1x1 + 5x5 convolution

                using (DeviceMemory<float> 
                    y1x1_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * InputInfo.SliceSize * OperationInfo.Primary5x5Reduce1x1ConvolutionKernels),
                    y_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * InputInfo.SliceSize * OperationInfo.Secondary5x5ConvolutionKernels))
                {
                    // 1x1 convolution
                    _5x5Reduce1x1OutputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, OperationInfo.Primary5x5Reduce1x1ConvolutionKernels, InputInfo.Height, InputInfo.Width);
                    DnnInstance.GetConvolutionForwardAlgorithm(InputDescription, _5x5Reduce1x1FilterDescription, _1x1ConvolutionDescription, _5x5Reduce1x1OutputDescription, ConvolutionFwdPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionFwdAlgo algorithm);
                    DnnInstance.GetConvolutionForwardWorkspaceSize(InputDescription, _5x5Reduce1x1FilterDescription, _1x1ConvolutionDescription, _5x5Reduce1x1OutputDescription, algorithm, out IntPtr size);
                    using (DeviceMemory<float> x_gpu = DnnInstance.Gpu.AllocateDevice(x))
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionForward(1, InputDescription, x_gpu.Ptr, _5x5Reduce1x1FilterDescription, pw_gpu += _3x3Weights, _1x1ConvolutionDescription, algorithm, workspace_gpu.Ptr, size, 0, _5x5Reduce1x1OutputDescription, y1x1_gpu.Ptr);                            
                    }
                    DnnInstance.AddTensor(1, _5x5Reduce1x1BiasDescription, pb_gpu += OperationInfo.Secondary3x3ConvolutionKernels, 1, _5x5Reduce1x1OutputDescription, y1x1_gpu.Ptr);
                    _5x5Reduce1x1Z.TryFree();
                    y1x1_gpu.CopyToHost(x.Entities, InputInfo.SliceSize * OperationInfo.Primary5x5Reduce1x1ConvolutionKernels, out _5x5Reduce1x1Z);
                    DnnInstance.ActivationForward(x.Entities, InputInfo.SliceSize * OperationInfo.Primary5x5Reduce1x1ConvolutionKernels, y1x1_gpu.Ptr, y1x1_gpu.Ptr, ActivationFunctions.Activation);
                    _5x5Reduce1x1A.TryFree();
                    y1x1_gpu.CopyToHost(x.Entities, InputInfo.SliceSize * OperationInfo.Primary5x5Reduce1x1ConvolutionKernels, out _5x5Reduce1x1A);

                    // 5x5 convolution
                    _5x5OutputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, OperationInfo.Secondary5x5ConvolutionKernels, InputInfo.Height, InputInfo.Width);
                    DnnInstance.GetConvolutionForwardAlgorithm(_5x5Reduce1x1OutputDescription, _5x5FilterDescription, _5x5ConvolutionDescription, _5x5OutputDescription, ConvolutionFwdPreference.PREFER_FASTEST, IntPtr.Zero, out algorithm);
                    DnnInstance.GetConvolutionForwardWorkspaceSize(_5x5Reduce1x1OutputDescription, _5x5FilterDescription, _5x5ConvolutionDescription, _5x5OutputDescription, algorithm, out size);
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionForward(1, _5x5Reduce1x1OutputDescription, y1x1_gpu.Ptr, _5x5FilterDescription, pw_gpu += _5x5Reduce1x1Weights, _5x5ConvolutionDescription, algorithm, workspace_gpu.Ptr, size, 0, _5x5OutputDescription, y_gpu.Ptr);
                    }
                    DnnInstance.AddTensor(1, _5x5BiasDescription, pb_gpu += OperationInfo.Primary5x5Reduce1x1ConvolutionKernels, 1, _5x5OutputDescription, y_gpu.Ptr);
                    y_gpu.CopyTo(z, InputInfo.SliceSize * (OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels), InputInfo.SliceSize * OperationInfo.Secondary5x5ConvolutionKernels);

                    // Activation
                    DnnInstance.ActivationForward(x.Entities, InputInfo.SliceSize * OperationInfo.Secondary5x5ConvolutionKernels, y_gpu.Ptr, y_gpu.Ptr, ActivationFunctions.Activation);
                    y_gpu.CopyTo(a, InputInfo.SliceSize * (OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels), InputInfo.SliceSize * OperationInfo.Secondary5x5ConvolutionKernels);
                }

                #endregion

                #region Pooling pipeline

                PoolingOutputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, InputInfo.Channels, InputInfo.Height, InputInfo.Width);
                using (DeviceMemory<float> y_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Size))
                {
                    // Pooling
                    using (DeviceMemory<float> x_gpu = DnnInstance.Gpu.AllocateDevice(x))
                    {
                        DnnInstance.PoolingForward(PoolingDescription, 1, InputDescription, x_gpu.Ptr, 0, InputDescription, y_gpu.Ptr);
                    }
                    _PoolingZ.TryFree();
                    y_gpu.CopyToHost(x.Entities, InputInfo.Size, out _PoolingZ);
                    DnnInstance.ActivationForward(x.Entities, x.Length, y_gpu.Ptr, y_gpu.Ptr, ActivationFunctions.Activation);
                    _PoolingA.TryFree();
                    y_gpu.CopyToHost(x.Entities, InputInfo.Size, out _PoolingA);

                    // 1x1 convolution
                    using (DeviceMemory<float> _1x1Output_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * InputInfo.SliceSize * OperationInfo.Secondary1x1AfterPoolingConvolutionKernels))
                    {
                        Secondary1x1OutputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, OperationInfo.Secondary1x1AfterPoolingConvolutionKernels, InputInfo.Height, InputInfo.Width);
                        DnnInstance.GetConvolutionForwardAlgorithm(InputDescription, Secondary1x1FilterDescription, _1x1ConvolutionDescription, Secondary1x1OutputDescription, ConvolutionFwdPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionFwdAlgo algorithm);
                        DnnInstance.GetConvolutionForwardWorkspaceSize(InputDescription, Secondary1x1FilterDescription, _1x1ConvolutionDescription, Secondary1x1OutputDescription, algorithm, out IntPtr size);
                        using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                        {
                            DnnInstance.ConvolutionForward(1, InputDescription, y_gpu.Ptr, Secondary1x1FilterDescription, pw_gpu + _5x5Weights, _1x1ConvolutionDescription, algorithm, workspace_gpu.Ptr, size, 0, Secondary1x1OutputDescription, _1x1Output_gpu.Ptr);                            
                        }
                        DnnInstance.AddTensor(1, Secondary1x1BiasDescription, pb_gpu + OperationInfo.Secondary5x5ConvolutionKernels, 1, Secondary1x1OutputDescription, _1x1Output_gpu.Ptr);
                        _1x1Output_gpu.CopyTo(z, InputInfo.SliceSize * (OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels + OperationInfo.Secondary5x5ConvolutionKernels), InputInfo.SliceSize * OperationInfo.Secondary1x1AfterPoolingConvolutionKernels);

                        // 1x1 convolution activation
                        DnnInstance.ActivationForward(x.Entities, InputInfo.SliceSize * OperationInfo.Secondary1x1AfterPoolingConvolutionKernels, _1x1Output_gpu.Ptr, _1x1Output_gpu.Ptr, ActivationFunctions.Activation);
                        _1x1Output_gpu.CopyTo(a, InputInfo.SliceSize * (OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels + OperationInfo.Secondary5x5ConvolutionKernels), InputInfo.SliceSize * OperationInfo.Secondary1x1AfterPoolingConvolutionKernels);
                    }
                }

                #endregion
            }
        }

        /// <inheritdoc/>
        public override void Backpropagate(in Tensor x, in Tensor y, in Tensor dy, in Tensor dx, out Tensor dJdw, out Tensor dJdb)
        {
            Tensor.New(1, Weights.Length, out dJdw);
            Tensor.New(1, Biases.Length, out dJdb);
            using (DeviceMemory<float>
                x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                dx_gpu = DnnInstance.Gpu.AllocateDevice<float>(dx.Size),
                w_gpu = DnnInstance.Gpu.AllocateDevice(Weights))
            {
                #region First 1x1 convolution

                // Backpropagation
                DnnInstance.GetConvolutionBackwardDataAlgorithm(_1x1FilterDescription, _1x1OutputDescription, _1x1ConvolutionDescription, InputDescription, ConvolutionBwdDataPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionBwdDataAlgo dAlgo);
                DnnInstance.GetConvolutionBackwardDataWorkspaceSize(_1x1FilterDescription, _1x1OutputDescription, _1x1ConvolutionDescription, InputDescription, dAlgo, out IntPtr size);
                using (DeviceMemory<float> 
                    _1x1dy_gpu = DnnInstance.Gpu.AllocateDevice(dy, 0, InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels),
                    _1x1y_gpu = DnnInstance.Gpu.AllocateDevice(y, 0, InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels))
                {
                    DnnInstance.ActivationBackward(y.Entities, InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels, _1x1y_gpu.Ptr, _1x1dy_gpu.Ptr, ActivationFunctions.ActivationPrime, _1x1dy_gpu.Ptr);
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionBackwardData(1, _1x1FilterDescription, w_gpu.Ptr, _1x1OutputDescription, _1x1dy_gpu.Ptr, _1x1ConvolutionDescription, dAlgo, workspace_gpu.Ptr, size, 0, InputDescription, dx_gpu.Ptr);
                    }

                    // Gradient
                    DnnInstance.GetConvolutionBackwardFilterAlgorithm(InputDescription, _1x1OutputDescription, _1x1ConvolutionDescription, _1x1FilterDescription, ConvolutionBwdFilterPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionBwdFilterAlgo wAlgo);
                    DnnInstance.GetConvolutionBackwardFilterWorkspaceSize(InputDescription, _1x1OutputDescription, _1x1ConvolutionDescription, _1x1FilterDescription, wAlgo, out size);
                    using (DeviceMemory<float> dw_gpu = DnnInstance.Gpu.AllocateDevice<float>(_1x1Weights))
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionBackwardFilter(1, InputDescription, x_gpu.Ptr, _1x1OutputDescription, _1x1dy_gpu.Ptr, _1x1ConvolutionDescription, wAlgo, workspace_gpu.Ptr, size, 0, _1x1FilterDescription, dw_gpu.Ptr);
                        dw_gpu.CopyTo(dJdw, 0, _1x1Weights);
                    }

                    // 1x1 bias
                    using (DeviceMemory<float> db_gpu = DnnInstance.Gpu.AllocateDevice<float>(OperationInfo.Primary1x1ConvolutionKernels))
                    {
                        DnnInstance.ConvolutionBackwardBias(1, _1x1OutputDescription, _1x1dy_gpu.Ptr, 0, _1x1BiasDescription, db_gpu.Ptr);
                        db_gpu.CopyTo(dJdb, 0, OperationInfo.Primary1x1ConvolutionKernels);
                    }
                }

                #endregion

                #region 1x1 + 3x3 convolution
                
                // 3x3 backward
                DnnInstance.GetConvolutionBackwardDataAlgorithm(_3x3FilterDescription, _3x3OutputDescription, _3x3ConvolutionDescription, _3x3Reduce1x1OutputDescription, ConvolutionBwdDataPreference.PREFER_FASTEST, IntPtr.Zero, out dAlgo);
                DnnInstance.GetConvolutionBackwardDataWorkspaceSize(_3x3FilterDescription, _3x3OutputDescription, _3x3ConvolutionDescription, _3x3Reduce1x1OutputDescription, dAlgo, out size);
                using (DeviceMemory<float> 
                    _3x3dy_gpu = DnnInstance.Gpu.AllocateDevice(dy, InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels, InputInfo.SliceSize * OperationInfo.Secondary3x3ConvolutionKernels),
                    _3x3y_gpu = DnnInstance.Gpu.AllocateDevice(y, InputInfo.SliceSize * OperationInfo.Primary1x1ConvolutionKernels, InputInfo.SliceSize * OperationInfo.Secondary3x3ConvolutionKernels),
                    _3x3Reduce1x1dx_gpu = DnnInstance.Gpu.AllocateDevice<float>(_3x3Reduce1x1Z.Size),
                    _3x3Reduce1x1z_gpu = DnnInstance.Gpu.AllocateDevice(_3x3Reduce1x1Z))
                {
                    DnnInstance.ActivationBackward(y.Entities, InputInfo.SliceSize * OperationInfo.Secondary3x3ConvolutionKernels, _3x3y_gpu.Ptr, _3x3dy_gpu.Ptr, ActivationFunctions.ActivationPrime, _3x3dy_gpu.Ptr);
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        deviceptr<float> p3x3Weights_gpu = w_gpu.Ptr + _1x1Weights + _3x3Reduce1x1Weights;
                        DnnInstance.ConvolutionBackwardData(1, _3x3FilterDescription, p3x3Weights_gpu, _3x3OutputDescription, _3x3dy_gpu.Ptr, _3x3ConvolutionDescription, dAlgo, workspace_gpu.Ptr, size, 0, _3x3Reduce1x1OutputDescription, _3x3Reduce1x1dx_gpu.Ptr);
                        DnnInstance.ActivationBackward(_3x3Reduce1x1Z.Entities, _3x3Reduce1x1Z.Length, _3x3Reduce1x1z_gpu.Ptr, _3x3Reduce1x1dx_gpu.Ptr, ActivationFunctions.ActivationPrime, _3x3Reduce1x1dx_gpu.Ptr);
                    }

                    // 3x3 gradient
                    DnnInstance.GetConvolutionBackwardFilterAlgorithm(_3x3Reduce1x1OutputDescription, _3x3OutputDescription, _3x3ConvolutionDescription, _3x3FilterDescription, ConvolutionBwdFilterPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionBwdFilterAlgo wAlgo);
                    DnnInstance.GetConvolutionBackwardFilterWorkspaceSize(_3x3Reduce1x1OutputDescription, _3x3OutputDescription, _3x3ConvolutionDescription, _3x3FilterDescription, wAlgo, out size);
                    using (DeviceMemory<float>
                        a3x3Reduce1x1_gpu = DnnInstance.Gpu.AllocateDevice(_3x3Reduce1x1A),
                        dw_gpu = DnnInstance.Gpu.AllocateDevice<float>(_3x3Weights))
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionBackwardFilter(1, _3x3Reduce1x1OutputDescription, a3x3Reduce1x1_gpu.Ptr, _3x3OutputDescription, _3x3dy_gpu.Ptr, _3x3ConvolutionDescription, wAlgo, workspace_gpu.Ptr, size, 0, _3x3FilterDescription, dw_gpu.Ptr);
                        dw_gpu.CopyTo(dJdw, _1x1Weights + _3x3Reduce1x1Weights, _3x3Weights);
                    }

                    // 3x3 bias
                    using (DeviceMemory<float> db_gpu = DnnInstance.Gpu.AllocateDevice<float>(OperationInfo.Secondary3x3ConvolutionKernels))
                    {
                        DnnInstance.ConvolutionBackwardBias(1, _3x3OutputDescription, _3x3dy_gpu.Ptr, 0, _3x3BiasDescription, db_gpu.Ptr);
                        db_gpu.CopyTo(dJdb, OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Primary3x3Reduce1x1ConvolutionKernels, OperationInfo.Secondary3x3ConvolutionKernels);
                    }

                    // 3x3 reduce 1x1 gradient
                    DnnInstance.GetConvolutionBackwardFilterAlgorithm(InputDescription, _3x3Reduce1x1OutputDescription, _1x1ConvolutionDescription, _3x3Reduce1x1FilterDescription, ConvolutionBwdFilterPreference.PREFER_FASTEST, IntPtr.Zero, out wAlgo);
                    DnnInstance.GetConvolutionBackwardFilterWorkspaceSize(InputDescription, _3x3Reduce1x1OutputDescription, _1x1ConvolutionDescription, _3x3Reduce1x1FilterDescription, wAlgo, out size);
                    using (DeviceMemory<float> dw_gpu = DnnInstance.Gpu.AllocateDevice<float>(_3x3Reduce1x1Weights))
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionBackwardFilter(1, InputDescription, x_gpu.Ptr, _3x3Reduce1x1OutputDescription, _3x3Reduce1x1dx_gpu.Ptr, _1x1ConvolutionDescription, wAlgo, workspace_gpu.Ptr, size, 0, _3x3Reduce1x1FilterDescription, dw_gpu.Ptr);
                        dw_gpu.CopyTo(dJdw, _1x1Weights, _3x3Reduce1x1Weights);
                    }

                    // 3x3 reduce 1x1 bias
                    using (DeviceMemory<float> db_gpu = DnnInstance.Gpu.AllocateDevice<float>(OperationInfo.Primary3x3Reduce1x1ConvolutionKernels))
                    {
                        DnnInstance.ConvolutionBackwardBias(1, _3x3Reduce1x1OutputDescription, _3x3Reduce1x1dx_gpu.Ptr, 0, _3x3Reduce1x1BiasDescription, db_gpu.Ptr);
                        db_gpu.CopyTo(dJdb, OperationInfo.Primary1x1ConvolutionKernels, OperationInfo.Primary3x3Reduce1x1ConvolutionKernels);
                    }

                    // 3x3 reduce 1x1 backward
                    DnnInstance.GetConvolutionBackwardDataAlgorithm(_3x3Reduce1x1FilterDescription, _3x3Reduce1x1OutputDescription, _1x1ConvolutionDescription, InputDescription, ConvolutionBwdDataPreference.PREFER_FASTEST, IntPtr.Zero, out dAlgo);
                    DnnInstance.GetConvolutionBackwardDataWorkspaceSize(_3x3Reduce1x1FilterDescription, _3x3Reduce1x1OutputDescription, _1x1ConvolutionDescription, InputDescription, dAlgo, out size);
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        deviceptr<float> p3x3Reduce1x1Weights_gpu = w_gpu.Ptr + _1x1Weights;
                        DnnInstance.ConvolutionBackwardData(1, _3x3Reduce1x1FilterDescription, p3x3Reduce1x1Weights_gpu, _3x3Reduce1x1OutputDescription, _3x3Reduce1x1dx_gpu.Ptr, _1x1ConvolutionDescription, dAlgo, workspace_gpu.Ptr, size, 1, InputDescription, dx_gpu.Ptr);
                    }
                }

                #endregion

                #region 1x1 + 5x5 convolution

                // 5x5 backward
                DnnInstance.GetConvolutionBackwardDataAlgorithm(_5x5FilterDescription, _5x5OutputDescription, _5x5ConvolutionDescription, _5x5Reduce1x1OutputDescription, ConvolutionBwdDataPreference.PREFER_FASTEST, IntPtr.Zero, out dAlgo);
                DnnInstance.GetConvolutionBackwardDataWorkspaceSize(_5x5FilterDescription, _5x5OutputDescription, _5x5ConvolutionDescription, _5x5Reduce1x1OutputDescription, dAlgo, out size);
                using (DeviceMemory<float> 
                    _5x5dy_gpu = DnnInstance.Gpu.AllocateDevice(dy, InputInfo.SliceSize * (OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels), InputInfo.SliceSize * OperationInfo.Secondary5x5ConvolutionKernels),
                    _5x5y_gpu = DnnInstance.Gpu.AllocateDevice(y, InputInfo.SliceSize * (OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels), InputInfo.SliceSize * OperationInfo.Secondary5x5ConvolutionKernels),
                    _5x5Reduce1x1dx_gpu = DnnInstance.Gpu.AllocateDevice<float>(_5x5Reduce1x1Z.Size),
                    _5x5Reduce1x1z_gpu = DnnInstance.Gpu.AllocateDevice(_5x5Reduce1x1Z))
                {
                    DnnInstance.ActivationBackward(y.Entities, InputInfo.SliceSize * OperationInfo.Secondary5x5ConvolutionKernels, _5x5y_gpu.Ptr, _5x5dy_gpu.Ptr, ActivationFunctions.ActivationPrime, _5x5dy_gpu.Ptr);
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        deviceptr<float> p5x5Weights_gpu = w_gpu.Ptr + _1x1Weights + _3x3Reduce1x1Weights + _3x3Weights + _5x5Reduce1x1Weights;
                        DnnInstance.ConvolutionBackwardData(1, _5x5FilterDescription, p5x5Weights_gpu, _5x5OutputDescription, _5x5dy_gpu.Ptr, _5x5ConvolutionDescription, dAlgo, workspace_gpu.Ptr, size, 0, _5x5Reduce1x1OutputDescription, _5x5Reduce1x1dx_gpu.Ptr);
                        DnnInstance.ActivationBackward(_5x5Reduce1x1Z.Entities, _5x5Reduce1x1Z.Length, _5x5Reduce1x1z_gpu.Ptr, _5x5Reduce1x1dx_gpu.Ptr, ActivationFunctions.ActivationPrime, _5x5Reduce1x1dx_gpu.Ptr);
                    }

                    // 5x5 gradient
                    DnnInstance.GetConvolutionBackwardFilterAlgorithm(_5x5Reduce1x1OutputDescription, _5x5OutputDescription, _5x5ConvolutionDescription, _5x5FilterDescription, ConvolutionBwdFilterPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionBwdFilterAlgo wAlgo);
                    DnnInstance.GetConvolutionBackwardFilterWorkspaceSize(_5x5Reduce1x1OutputDescription, _5x5OutputDescription, _5x5ConvolutionDescription, _5x5FilterDescription, wAlgo, out size);
                    using (DeviceMemory<float>
                        a5x5Reduce1x1_gpu = DnnInstance.Gpu.AllocateDevice(_5x5Reduce1x1A),
                        dw_gpu = DnnInstance.Gpu.AllocateDevice<float>(_5x5Weights))
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionBackwardFilter(1, _5x5Reduce1x1OutputDescription, a5x5Reduce1x1_gpu.Ptr, _5x5OutputDescription, _5x5dy_gpu.Ptr, _5x5ConvolutionDescription, wAlgo, workspace_gpu.Ptr, size, 0, _5x5FilterDescription, dw_gpu.Ptr);
                        dw_gpu.CopyTo(dJdw, _1x1Weights + _3x3Reduce1x1Weights + _3x3Weights + _5x5Reduce1x1Weights, _5x5Weights);
                    }

                    // 5x5 bias
                    using (DeviceMemory<float> db_gpu = DnnInstance.Gpu.AllocateDevice<float>(OperationInfo.Secondary5x5ConvolutionKernels))
                    {
                        DnnInstance.ConvolutionBackwardBias(1, _5x5OutputDescription, _5x5dy_gpu.Ptr, 0, _5x5BiasDescription, db_gpu.Ptr);
                        db_gpu.CopyTo(dJdb, OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Primary3x3Reduce1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels + OperationInfo.Primary5x5Reduce1x1ConvolutionKernels, OperationInfo.Secondary5x5ConvolutionKernels);
                    }

                    // 5x5 reduce 1x1 weights
                    DnnInstance.GetConvolutionBackwardFilterAlgorithm(InputDescription, _5x5Reduce1x1OutputDescription, _1x1ConvolutionDescription, _5x5Reduce1x1FilterDescription, ConvolutionBwdFilterPreference.PREFER_FASTEST, IntPtr.Zero, out wAlgo);
                    DnnInstance.GetConvolutionBackwardFilterWorkspaceSize(InputDescription, _5x5Reduce1x1OutputDescription, _1x1ConvolutionDescription, _5x5Reduce1x1FilterDescription, wAlgo, out size);
                    using (DeviceMemory<float> dw_gpu = DnnInstance.Gpu.AllocateDevice<float>(_5x5Reduce1x1Weights))
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionBackwardFilter(1, InputDescription, x_gpu.Ptr, _5x5Reduce1x1OutputDescription, _5x5Reduce1x1dx_gpu.Ptr, _1x1ConvolutionDescription, wAlgo, workspace_gpu.Ptr, size, 0, _5x5Reduce1x1FilterDescription, dw_gpu.Ptr);
                        dw_gpu.CopyTo(dJdw, _1x1Weights + _3x3Reduce1x1Weights + _3x3Weights, _5x5Reduce1x1Weights);
                    }

                    // 5x5 reduce 1x1 bias
                    using (DeviceMemory<float> db_gpu = DnnInstance.Gpu.AllocateDevice<float>(OperationInfo.Primary5x5Reduce1x1ConvolutionKernels))
                    {
                        DnnInstance.ConvolutionBackwardBias(1, _5x5Reduce1x1OutputDescription, _5x5Reduce1x1dx_gpu.Ptr, 0, _5x5Reduce1x1BiasDescription, db_gpu.Ptr);
                        db_gpu.CopyTo(dJdb, OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Primary3x3Reduce1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels, OperationInfo.Primary5x5Reduce1x1ConvolutionKernels);
                    }

                    // 5x5 reduce 1x1 backward
                    DnnInstance.GetConvolutionBackwardDataAlgorithm(_5x5Reduce1x1FilterDescription, _5x5Reduce1x1OutputDescription, _1x1ConvolutionDescription, InputDescription, ConvolutionBwdDataPreference.PREFER_FASTEST, IntPtr.Zero, out dAlgo);
                    DnnInstance.GetConvolutionBackwardDataWorkspaceSize(_5x5Reduce1x1FilterDescription, _5x5Reduce1x1OutputDescription, _1x1ConvolutionDescription, InputDescription, dAlgo, out size);
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        deviceptr<float> p5x5Reduce1x1Weights_gpu = w_gpu.Ptr + _1x1Weights + _3x3Reduce1x1Weights + _3x3Weights;
                        DnnInstance.ConvolutionBackwardData(1, _5x5Reduce1x1FilterDescription, p5x5Reduce1x1Weights_gpu, _5x5Reduce1x1OutputDescription, _5x5Reduce1x1dx_gpu.Ptr, _1x1ConvolutionDescription, dAlgo, workspace_gpu.Ptr, size, 1, InputDescription, dx_gpu.Ptr);
                    }
                }

                #endregion

                #region Pooling

                using (DeviceMemory<float> pooly_gpu = DnnInstance.Gpu.AllocateDevice(_PoolingZ))
                {
                    // 1x1 backward
                    using (DeviceMemory<float> 
                        dy_gpu = DnnInstance.Gpu.AllocateDevice(dy, InputInfo.SliceSize * (OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels + OperationInfo.Secondary5x5ConvolutionKernels), InputInfo.SliceSize * OperationInfo.Secondary1x1AfterPoolingConvolutionKernels),
                        y_gpu = DnnInstance.Gpu.AllocateDevice(y, InputInfo.SliceSize * (OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels + OperationInfo.Secondary5x5ConvolutionKernels), InputInfo.SliceSize * OperationInfo.Secondary1x1AfterPoolingConvolutionKernels),
                        pooldy_gpu = DnnInstance.Gpu.AllocateDevice<float>(_PoolingZ.Size))
                    {
                        DnnInstance.ActivationBackward(y.Entities, InputInfo.SliceSize * OperationInfo.Secondary1x1AfterPoolingConvolutionKernels, y_gpu.Ptr, dy_gpu.Ptr, ActivationFunctions.ActivationPrime, dy_gpu.Ptr);
                        DnnInstance.GetConvolutionBackwardDataAlgorithm(Secondary1x1FilterDescription, Secondary1x1OutputDescription, _1x1ConvolutionDescription, PoolingOutputDescription, ConvolutionBwdDataPreference.PREFER_FASTEST, IntPtr.Zero, out dAlgo);
                        DnnInstance.GetConvolutionBackwardDataWorkspaceSize(Secondary1x1FilterDescription, Secondary1x1OutputDescription, _1x1ConvolutionDescription, PoolingOutputDescription, dAlgo, out size);
                        using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                        {
                            deviceptr<float> p1x1PoolingWeights_gpu = w_gpu.Ptr + _1x1Weights + _3x3Reduce1x1Weights + _3x3Weights + _5x5Reduce1x1Weights + _5x5Weights;
                            DnnInstance.ConvolutionBackwardData(1, Secondary1x1FilterDescription, p1x1PoolingWeights_gpu, Secondary1x1OutputDescription, dy_gpu.Ptr, _1x1ConvolutionDescription, dAlgo, workspace_gpu.Ptr, size, 0, PoolingOutputDescription, pooldy_gpu.Ptr);
                            DnnInstance.ActivationBackward(_PoolingZ.Entities, _PoolingZ.Length, pooly_gpu.Ptr, pooldy_gpu.Ptr, ActivationFunctions.ActivationPrime, pooldy_gpu.Ptr);
                        }
                        
                        // Pooling backward
                        DnnInstance.PoolingBackward(PoolingDescription, 1, PoolingOutputDescription, pooly_gpu.Ptr, PoolingOutputDescription, pooldy_gpu.Ptr, InputDescription, x_gpu.Ptr, 1, InputDescription, dx_gpu.Ptr);

                        // 1x1 gradient
                        DnnInstance.GetConvolutionBackwardFilterAlgorithm(PoolingOutputDescription, Secondary1x1OutputDescription, _1x1ConvolutionDescription, Secondary1x1FilterDescription, ConvolutionBwdFilterPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionBwdFilterAlgo wAlgo);
                        DnnInstance.GetConvolutionBackwardFilterWorkspaceSize(PoolingOutputDescription, Secondary1x1OutputDescription, _1x1ConvolutionDescription, Secondary1x1FilterDescription, wAlgo, out size);
                        using (DeviceMemory<float>
                            aPool_gpu = DnnInstance.Gpu.AllocateDevice(_PoolingA),
                            dw_gpu = DnnInstance.Gpu.AllocateDevice<float>(Secondary1x1Weights))
                        using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                        {
                            DnnInstance.ConvolutionBackwardFilter(1, PoolingOutputDescription, aPool_gpu.Ptr, Secondary1x1OutputDescription, dy_gpu.Ptr, _1x1ConvolutionDescription, wAlgo, workspace_gpu.Ptr, size, 0, Secondary1x1FilterDescription, dw_gpu.Ptr);
                            dw_gpu.CopyTo(dJdw, _1x1Weights + _3x3Reduce1x1Weights + _3x3Weights + _5x5Reduce1x1Weights + _5x5Weights, Secondary1x1Weights);
                        }

                        // Pooling 1x1 bias
                        using (DeviceMemory<float> db_gpu = DnnInstance.Gpu.AllocateDevice<float>(OperationInfo.Secondary1x1AfterPoolingConvolutionKernels))
                        {
                            DnnInstance.ConvolutionBackwardBias(1, Secondary1x1OutputDescription, dy_gpu.Ptr, 0, Secondary1x1BiasDescription, db_gpu.Ptr);
                            db_gpu.CopyTo(dJdb, OperationInfo.Primary1x1ConvolutionKernels + OperationInfo.Primary3x3Reduce1x1ConvolutionKernels + OperationInfo.Secondary3x3ConvolutionKernels + OperationInfo.Primary5x5Reduce1x1ConvolutionKernels + OperationInfo.Secondary5x5ConvolutionKernels, OperationInfo.Secondary1x1AfterPoolingConvolutionKernels);
                        }
                    }
                }

                #endregion

                // Copy the gradient back to RAM
                dx_gpu.CopyTo(dx);
            }
        }

        #endregion

        #region Misc

        /// <inheritdoc/>
        public override void Serialize(System.IO.Stream stream)
        {
            base.Serialize(stream);
            stream.Write(OperationInfo);
        }

        /// <summary>
        /// Tries to deserialize a new <see cref="CuDnnInceptionLayer"/> from the input <see cref="System.IO.Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="System.IO.Stream"/> to use to read the layer data</param>
        [MustUseReturnValue, CanBeNull]
        public static INetworkLayer Deserialize([NotNull] System.IO.Stream stream)
        {
            if (!stream.TryRead(out TensorInfo input)) return null;
            if (!stream.TryRead<TensorInfo>(out _)) return null;
            if (!stream.TryRead<ActivationType>(out _)) return null;
            if (!stream.TryRead(out int wLength)) return null;
            float[] weights = stream.ReadUnshuffled(wLength);
            if (!stream.TryRead(out int bLength)) return null;
            float[] biases = stream.ReadUnshuffled(bLength);
            if (!stream.TryRead(out InceptionInfo info)) return null;
            return new CuDnnInceptionLayer(input, info, weights, biases);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new CuDnnInceptionLayer(InputInfo, OperationInfo, Weights, Biases);

        #endregion

        #region IDisposable

        ~CuDnnInceptionLayer() => Dispose();

        /// <inheritdoc/>
        void IDisposable.Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose();
        }

        // Private Dispose method
        private void Dispose()
        {
            _3x3Reduce1x1Z.TryFree();
            _3x3Reduce1x1A.TryFree();
            _5x5Reduce1x1Z.TryFree();
            _5x5Reduce1x1A.TryFree();
            _PoolingZ.TryFree();
            _PoolingA.TryFree();
        }

        #endregion
    }
}
