using System;
using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.Cuda.Services;
using NeuralNetworkNET.Cuda.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.Cuda.Layers
{
    /// <summary>
    /// A 2D convolutional layer based on the cuDNN back-end
    /// </summary>
    internal sealed class CuDnnConvolutionalLayer : ConvolutionalLayer
    {
        #region cuDNN fields

        // The NCHW tensor info for the layer inputs
        [NotNull]
        private readonly TensorDescriptor InputDescription = new TensorDescriptor();

        // The NCHW info for the layer weights
        [NotNull]
        private readonly FilterDescriptor FilterDescription = new FilterDescriptor();

        // The info on the layer bias (one value per output channel)
        [NotNull]
        private readonly TensorDescriptor BiasDescription = new TensorDescriptor();

        // The convolution info
        [NotNull]
        private readonly ConvolutionDescriptor ConvolutionDescription = new ConvolutionDescriptor();

        // The NCHW tensor info for the layer outputs
        [NotNull]
        private readonly TensorDescriptor OutputDescription = new TensorDescriptor();

        /// <summary>
        /// Gets the <see cref="Dnn"/> instance for the current layer
        /// </summary>
        [NotNull]
        private readonly Dnn DnnInstance = DnnService.Instance;

        /// <summary>
        /// Sets the cuDNN fields that will be used during future forward/backwards operations
        /// </summary>
        private void SetupCuDnnInfo()
        {
            ConvolutionDescription.Set2D(OperationInfo.VerticalPadding, OperationInfo.HorizontalPadding, OperationInfo.VerticalStride, OperationInfo.HorizontalStride, 1, 1, (Alea.cuDNN.ConvolutionMode)OperationInfo.Mode);
            FilterDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, OutputInfo.Channels, KernelInfo.Channels, KernelInfo.Height, KernelInfo.Width);
            BiasDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, 1, OutputInfo.Channels, 1, 1);
        }

        #endregion

        public CuDnnConvolutionalLayer(
            in TensorInfo input, in ConvolutionInfo operation, (int X, int Y) kernelSize, int kernels,
            ActivationFunctionType activation, BiasInitializationMode biasMode)
            : base(input, operation, kernelSize, kernels, activation, biasMode)
            => SetupCuDnnInfo();

        public CuDnnConvolutionalLayer(
            in TensorInfo input, in ConvolutionInfo operation, TensorInfo kernels, TensorInfo output,
            [NotNull] float[,] weights, [NotNull] float[] biases, ActivationFunctionType activation)
            : base(input, operation, kernels, output, weights, biases, activation)
            => SetupCuDnnInfo();

        #region Implementation

        /// <inheritdoc/>
        public override unsafe void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            fixed (float* pw = Weights)
            {
                Tensor.Fix(pw, OutputInfo.Channels, KernelInfo.Size, out Tensor wTensor);
                using (DeviceMemory<float> z_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * OutputInfo.Size))
                {
                    // Tensors info setup
                    InputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, InputInfo.Channels, InputInfo.Height, InputInfo.Width);
                    OutputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, OutputInfo.Channels, OutputInfo.Height, OutputInfo.Width);

                    // Forward convolution
                    DnnInstance.GetConvolutionForwardAlgorithm(InputDescription, FilterDescription, ConvolutionDescription, OutputDescription, ConvolutionFwdPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionFwdAlgo algorithm);
                    DnnInstance.GetConvolutionForwardWorkspaceSize(InputDescription, FilterDescription, ConvolutionDescription, OutputDescription, algorithm, out IntPtr size);
                    using (DeviceMemory<float>
                        x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                        w_gpu = DnnInstance.Gpu.AllocateDevice(wTensor))
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionForward(1, InputDescription, x_gpu.Ptr, FilterDescription, w_gpu.Ptr, ConvolutionDescription, algorithm, workspace_gpu.Ptr, size, 0, OutputDescription, z_gpu.Ptr);
                    }

                    // Biases
                    using (DeviceMemory<float> b_gpu = DnnInstance.Gpu.AllocateDevice(Biases))
                    {
                        DnnInstance.AddTensor(1, BiasDescription, b_gpu.Ptr, 1, OutputDescription, z_gpu.Ptr);
                    }
                    z_gpu.CopyToHost(x.Entities, OutputInfo.Size, out z);

                    // Activation
                    if (ActivationFunctionType == ActivationFunctionType.Identity) z.Duplicate(out a);
                    else
                    {
                        DnnInstance.ActivationForward(z.Entities, z.Length, z_gpu.Ptr, z_gpu.Ptr, ActivationFunctions.Activation);
                        z_gpu.CopyToHost(z.Entities, z.Length, out a);
                    }
                }
            }
        }

        /// <inheritdoc/>
        public override unsafe void Backpropagate(in Tensor delta_1, in Tensor z, ActivationFunction activationPrime)
        {
            fixed (float* pw = Weights)
            {
                Tensor.Fix(pw, OutputInfo.Channels, KernelInfo.Size, out Tensor wTensor);
                DnnInstance.GetConvolutionBackwardDataAlgorithm(FilterDescription, OutputDescription, ConvolutionDescription, InputDescription, ConvolutionBwdDataPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionBwdDataAlgo algorithm);
                DnnInstance.GetConvolutionBackwardDataWorkspaceSize(FilterDescription, OutputDescription, ConvolutionDescription, InputDescription, algorithm, out IntPtr size);
                using (DeviceMemory<float> delta_gpu = DnnInstance.Gpu.AllocateDevice<float>(z.Size))
                {
                    // Backwards convolution
                    using (DeviceMemory<float>
                        delta_1_gpu = DnnInstance.Gpu.AllocateDevice(delta_1),
                        w_gpu = DnnInstance.Gpu.AllocateDevice(wTensor))
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionBackwardData(1, FilterDescription, w_gpu.Ptr, OutputDescription, delta_1_gpu.Ptr, ConvolutionDescription, algorithm, workspace_gpu.Ptr, size, 0, InputDescription, delta_gpu.Ptr);
                    }

                    // Activation
                    using (DeviceMemory<float> z_gpu = DnnInstance.Gpu.AllocateDevice(z))
                    {
                        DnnInstance.ActivationBackward(z.Entities, z.Length, z_gpu.Ptr, delta_gpu.Ptr, activationPrime);
                        z_gpu.CopyTo(z);
                    }
                }
            }
        }

        /// <inheritdoc/>
        public override void ComputeGradient(in Tensor a, in Tensor delta, out Tensor dJdw, out Tensor dJdb)
        {
            using (DeviceMemory<float> delta_gpu = DnnInstance.Gpu.AllocateDevice(delta))
            {
                // Kernels
                using (DeviceMemory<float>
                    a_gpu = DnnInstance.Gpu.AllocateDevice(a),
                    w_gpu = DnnInstance.Gpu.AllocateDevice<float>(Kernels * KernelInfo.Size))
                {
                    DnnInstance.GetConvolutionBackwardFilterAlgorithm(InputDescription, OutputDescription, ConvolutionDescription, FilterDescription, ConvolutionBwdFilterPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionBwdFilterAlgo algorithm);
                    DnnInstance.GetConvolutionBackwardFilterWorkspaceSize(InputDescription, OutputDescription, ConvolutionDescription, FilterDescription, algorithm, out IntPtr size);
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionBackwardFilter(1, InputDescription, a_gpu.Ptr, OutputDescription, delta_gpu.Ptr, ConvolutionDescription, algorithm, workspace_gpu.Ptr, size, 0, FilterDescription, w_gpu.Ptr);
                    }
                    w_gpu.CopyToHost(Kernels, KernelInfo.Size, out dJdw);
                }

                // Bias
                using (DeviceMemory<float> dJdb_gpu = DnnInstance.Gpu.AllocateDevice<float>(Biases.Length))
                {
                    DnnInstance.ConvolutionBackwardBias(1, OutputDescription, delta_gpu.Ptr, 0, BiasDescription, dJdb_gpu.Ptr);
                    dJdb_gpu.CopyToHost(1, OutputInfo.Channels, out dJdb);
                }
            }
        }

        #endregion

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new CuDnnConvolutionalLayer(InputInfo, OperationInfo, KernelInfo, OutputInfo, Weights.BlockCopy(), Biases.BlockCopy(), ActivationFunctionType);
    }
}