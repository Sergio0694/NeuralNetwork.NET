using System;
using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Layers.Cpu;

namespace NeuralNetworkNET.Networks.Layers.Cuda
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
        private readonly Dnn DnnInstance = CuDnnService.Instance;

        // cuDNN fields setup
        private void SetupCuDnnInfo()
        {
            ConvolutionDescription.Set2D(OperationInfo.VerticalPadding, OperationInfo.HorizontalPadding, OperationInfo.VerticalStride, OperationInfo.HorizontalStride, 1, 1, (Alea.cuDNN.ConvolutionMode)OperationInfo.Mode);
            FilterDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, OutputInfo.Channels, KernelInfo.Channels, KernelInfo.Height, KernelInfo.Width);
            BiasDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, 1, OutputInfo.Channels, 1, 1);
        }

        #endregion

        public CuDnnConvolutionalLayer(
            in TensorInfo input, in ConvolutionInfo operation, (int X, int Y) kernelSize, int kernels,
            ActivationType activation, BiasInitializationMode biasMode)
            : base(input, operation, kernelSize, kernels, activation, biasMode)
            => SetupCuDnnInfo();

        public CuDnnConvolutionalLayer(
            in TensorInfo input, in ConvolutionInfo operation, TensorInfo kernels, TensorInfo output,
            [NotNull] float[] weights, [NotNull] float[] biases, ActivationType activation)
            : base(input, operation, kernels, output, weights, biases, activation)
            => SetupCuDnnInfo();

        #region Implementation

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
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
                    w_gpu = DnnInstance.Gpu.AllocateDevice(Weights))
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
                if (ActivationType == ActivationType.Identity) z.Duplicate(out a);
                else
                {
                    DnnInstance.ActivationForward(z.Entities, z.Length, z_gpu.Ptr, z_gpu.Ptr, ActivationFunctions.Activation);
                    z_gpu.CopyToHost(z.Entities, z.Length, out a);
                }
            }
        }

        /// <inheritdoc/>
        public override void Backpropagate(in Tensor x, in Tensor y, in Tensor dy, in Tensor dx, out Tensor dJdw, out Tensor dJdb)
        {
            using (DeviceMemory<float>
                x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                y_gpu = DnnInstance.Gpu.AllocateDevice(y),
                dy_gpu = DnnInstance.Gpu.AllocateDevice(dy),
                w_gpu = DnnInstance.Gpu.AllocateDevice(Weights),
                dJdb_gpu = DnnInstance.Gpu.AllocateDevice<float>(Biases.Length))
            {
                // Backpropagation
                DnnInstance.ActivationBackward(y.Entities, y.Length, y_gpu.Ptr, dy_gpu.Ptr, ActivationFunctions.ActivationPrime, dy_gpu.Ptr);
                if (!dx.IsNull)
                {
                    DnnInstance.GetConvolutionBackwardDataAlgorithm(FilterDescription, OutputDescription, ConvolutionDescription, InputDescription, ConvolutionBwdDataPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionBwdDataAlgo dAlgo);
                    DnnInstance.GetConvolutionBackwardDataWorkspaceSize(FilterDescription, OutputDescription, ConvolutionDescription, InputDescription, dAlgo, out IntPtr dSize);
                    using (DeviceMemory<float> dx_gpu = DnnInstance.Gpu.AllocateDevice<float>(dx.Size))
                    using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(dSize))
                    {
                        DnnInstance.ConvolutionBackwardData(1, FilterDescription, w_gpu.Ptr, OutputDescription, dy_gpu.Ptr, ConvolutionDescription, dAlgo, workspace_gpu.Ptr, dSize, 0, InputDescription, dx_gpu.Ptr);
                        dx_gpu.CopyTo(dx);
                    }
                }

                // Gradient
                DnnInstance.GetConvolutionBackwardFilterAlgorithm(InputDescription, OutputDescription, ConvolutionDescription, FilterDescription, ConvolutionBwdFilterPreference.PREFER_FASTEST, IntPtr.Zero, out ConvolutionBwdFilterAlgo wAlgo);
                DnnInstance.GetConvolutionBackwardFilterWorkspaceSize(InputDescription, OutputDescription, ConvolutionDescription, FilterDescription, wAlgo, out IntPtr wSize);
                using (DeviceMemory<byte> workspace_gpu = DnnInstance.Gpu.AllocateDevice<byte>(wSize))
                {
                    DnnInstance.ConvolutionBackwardFilter(1, InputDescription, x_gpu.Ptr, OutputDescription, dy_gpu.Ptr, ConvolutionDescription, wAlgo, workspace_gpu.Ptr, wSize, 0, FilterDescription, w_gpu.Ptr);
                }
                w_gpu.CopyToHost(1, Weights.Length, out dJdw);
                DnnInstance.ConvolutionBackwardBias(1, OutputDescription, dy_gpu.Ptr, 0, BiasDescription, dJdb_gpu.Ptr);
                dJdb_gpu.CopyToHost(1, OutputInfo.Channels, out dJdb);
            }
        }

        #endregion

        #region Misc

        /// <summary>
        /// Tries to deserialize a new <see cref="CuDnnConvolutionalLayer"/> from the input <see cref="System.IO.Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="System.IO.Stream"/> to use to read the layer data</param>
        [MustUseReturnValue, CanBeNull]
        public new static INetworkLayer Deserialize([NotNull] System.IO.Stream stream)
        {
            if (!stream.TryRead(out TensorInfo input)) return null;
            if (!stream.TryRead(out TensorInfo output)) return null;
            if (!stream.TryRead(out ActivationType activation)) return null;
            if (!stream.TryRead(out int wLength)) return null;
            float[] weights = stream.ReadUnshuffled(wLength);
            if (!stream.TryRead(out int bLength)) return null;
            float[] biases = stream.ReadUnshuffled(bLength);
            if (!stream.TryRead(out ConvolutionInfo operation)) return null;
            if (!stream.TryRead(out TensorInfo kernels)) return null;
            return new CuDnnConvolutionalLayer(input, operation, kernels, output, weights, biases, activation);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new CuDnnConvolutionalLayer(InputInfo, OperationInfo, KernelInfo, OutputInfo, Weights.AsSpan().ToArray(), Biases.AsSpan().ToArray(), ActivationType);

        #endregion
    }
}