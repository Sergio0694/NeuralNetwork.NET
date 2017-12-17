using System;
using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Misc;
using NeuralNetworkNET.Cuda.Services;
using NeuralNetworkNET.Cuda.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers;
using NeuralNetworkNET.Structs;

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
        private readonly TensorDescriptor InputInfo = new TensorDescriptor();

        // The NCHW info for the layer weights
        [NotNull]
        private readonly FilterDescriptor FilterInfo = new FilterDescriptor();

        // The info on the layer bias (one value per output channel)
        [NotNull]
        private readonly TensorDescriptor BiasInfo = new TensorDescriptor();

        // The convolution info
        [NotNull]
        private readonly ConvolutionDescriptor ConvolutionInfo = new ConvolutionDescriptor();

        // The NCHW tensor info for the layer outputs
        [NotNull]
        private readonly TensorDescriptor OutputInfo = new TensorDescriptor();

        /// <summary>
        /// Gets the <see cref="Dnn"/> instance for the current layer
        /// </summary>
        [NotNull]
        private readonly Dnn DnnInstance = DnnService.Instance;

        /// <summary>
        /// Sets the cuDNN fields that will be used during future forward/backwards operations
        /// </summary>
        /// <param name="mode">The desired convolution mode</param>
        private void SetupCuDnnInfo(ConvolutionMode mode)
        {
            ConvolutionInfo.Set2D(0, 0, 1, 1, 1, 1, mode);
            FilterInfo.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, OutputVolume.Depth, KernelVolume.Depth, KernelVolume.Height, KernelVolume.Width);
            BiasInfo.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, 1, OutputVolume.Depth, 1, 1);
        }

        #endregion

        public CuDnnConvolutionalLayer(
            VolumeInformation input, (int X, int Y) kernelSize, int kernels,
            ActivationFunctionType activation, ConvolutionMode mode)
            : base(input, kernelSize, kernels, activation)
            => SetupCuDnnInfo(mode);

        public CuDnnConvolutionalLayer(
            VolumeInformation input, VolumeInformation kernels, VolumeInformation output,
            [NotNull] float[,] weights, [NotNull] float[] biases,
            ActivationFunctionType activation, ConvolutionMode mode)
            : base(input, kernels, output, weights, biases, activation)
            => SetupCuDnnInfo(mode);

        /// <inheritdoc/>
        public override unsafe void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            fixed (float* pw = Weights)
            {
                Tensor.Fix(pw, OutputVolume.Depth, KernelVolume.Volume, out Tensor wSpan);
                using (DeviceMemory<float> z_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * OutputVolume.Volume))
                {
                    // Tensors info setup
                    InputInfo.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, InputVolume.Depth, InputVolume.Height, InputVolume.Width);
                    OutputInfo.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, OutputVolume.Depth, OutputVolume.Height, OutputVolume.Width);

                    // Forward convolution
                    DnnInstance.GetConvolutionForwardAlgorithm(InputInfo, FilterInfo, ConvolutionInfo, OutputInfo, ConvolutionFwdPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionFwdAlgo algorithm);
                    DnnInstance.GetConvolutionForwardWorkspaceSize(InputInfo, FilterInfo, ConvolutionInfo, OutputInfo, algorithm, out IntPtr size);
                    using (DeviceMemory<float>
                        x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                        w_gpu = DnnInstance.Gpu.AllocateDevice(wSpan))
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionForward(1, InputInfo, x_gpu.Ptr, FilterInfo, w_gpu.Ptr, ConvolutionInfo, algorithm, workspace_gpu.Ptr, size, 0, OutputInfo, z_gpu.Ptr);
                    }

                    // Biases
                    using (DeviceMemory<float> b_gpu = DnnInstance.Gpu.AllocateDevice(Biases))
                    {
                        DnnInstance.AddTensor(1, BiasInfo, b_gpu.Ptr, 1, OutputInfo, z_gpu.Ptr);
                    }
                    z_gpu.CopyToHost(x.Entities, OutputVolume.Volume, out z);

                    // Activation
                    if (ActivationFunctionType == ActivationFunctionType.Identity) z.Duplicate(out a);
                    else
                    {
                        DnnInstance.ActivationForward(z.Entities, z.Length, z_gpu.Ptr, z.Length, z_gpu.Ptr, z.Length, ActivationFunctions.Activation);
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
                Tensor.Fix(pw, OutputVolume.Depth, KernelVolume.Volume, out Tensor wSpan);
                DnnInstance.GetConvolutionBackwardDataAlgorithm(FilterInfo, OutputInfo, ConvolutionInfo, InputInfo, ConvolutionBwdDataPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionBwdDataAlgo algorithm);
                DnnInstance.GetConvolutionBackwardDataWorkspaceSize(FilterInfo, OutputInfo, ConvolutionInfo, InputInfo, algorithm, out IntPtr size);
                using (DeviceMemory<float> delta_gpu = DnnInstance.Gpu.AllocateDevice<float>(z.Size))
                {
                    // Backwards convolution
                    using (DeviceMemory<float>
                        delta_1_gpu = DnnInstance.Gpu.AllocateDevice(delta_1),
                        w_gpu = DnnInstance.Gpu.AllocateDevice(wSpan))
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionBackwardData(1, FilterInfo, w_gpu.Ptr, OutputInfo, delta_1_gpu.Ptr, ConvolutionInfo, algorithm, workspace_gpu.Ptr, size, 0, InputInfo, delta_gpu.Ptr);
                    }

                    // Activation
                    using (DeviceMemory<float> z_gpu = DnnInstance.Gpu.AllocateDevice(z))
                    {
                        DnnInstance.ActivationBackward(z.Entities, z.Length, z_gpu.Ptr, z.Length, delta_gpu.Ptr, z.Length, activationPrime);
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
                    w_gpu = DnnInstance.Gpu.AllocateDevice<float>(Kernels * KernelVolume.Volume))
                {
                    DnnInstance.GetConvolutionBackwardFilterAlgorithm(InputInfo, OutputInfo, ConvolutionInfo, FilterInfo, ConvolutionBwdFilterPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionBwdFilterAlgo algorithm);
                    DnnInstance.GetConvolutionBackwardFilterWorkspaceSize(InputInfo, OutputInfo, ConvolutionInfo, FilterInfo, algorithm, out IntPtr size);
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(size))
                    {
                        DnnInstance.ConvolutionBackwardFilter(1, InputInfo, a_gpu.Ptr, OutputInfo, delta_gpu.Ptr, ConvolutionInfo, algorithm, workspace_gpu.Ptr, size, 0, FilterInfo, w_gpu.Ptr);
                    }
                    w_gpu.CopyToHost(Kernels, KernelVolume.Volume, out dJdw);
                }

                // Bias
                using (DeviceMemory<float> dJdb_gpu = DnnInstance.Gpu.AllocateDevice<float>(Biases.Length))
                {
                    DnnInstance.ConvolutionBackwardBias(1, OutputInfo, delta_gpu.Ptr, 0, BiasInfo, dJdb_gpu.Ptr);
                    dJdb_gpu.CopyToHost(1, OutputVolume.Depth, out dJdb);
                }
            }
        }
    }
}