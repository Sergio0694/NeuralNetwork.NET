using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.Cuda.Services;
using NeuralNetworkNET.Cuda.Extensions;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers;
using NeuralNetworkNET.Structs;
using NeuralNetworkNET.APIs.Misc;

namespace NeuralNetworkNET.Cuda.Layers
{
    internal class CuDnnFullyConnectedLayer : FullyConnectedLayer
    {
        /// <summary>
        /// Gets the <see cref="Dnn"/> instance for the current layer
        /// </summary>
        [NotNull]
        private readonly Dnn DnnInstance = DnnService.Instance;

        public CuDnnFullyConnectedLayer(in TensorInfo input, int outputs, ActivationFunctionType activation) 
            : base(input, outputs, activation) { }

        public CuDnnFullyConnectedLayer([NotNull] float[,] weights, [NotNull] float[] biases, ActivationFunctionType activation) 
            : base(weights, biases, activation) { }

        /// <inheritdoc/>
        public override unsafe void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            fixed (float* pw = Weights)
            {
                Tensor.Fix(pw, InputInfo.Size, OutputInfo.Size, out Tensor wSpan);
                using (DeviceMemory<float>
                    x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                    w_gpu = DnnInstance.Gpu.AllocateDevice(wSpan),
                    y_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * OutputInfo.Size),
                    b_gpu = DnnInstance.Gpu.AllocateDevice(Biases))
                {
                    DnnInstance.FullyConnectedForward(x.Entities, x.Length, OutputInfo.Size, x_gpu.Ptr, x.Length, w_gpu.Ptr, wSpan.Length, b_gpu.Ptr, y_gpu.Ptr, OutputInfo.Size);
                    y_gpu.CopyToHost(x.Entities, OutputInfo.Size, out z);
                    DnnInstance.ActivationForward(z.Entities, z.Length, y_gpu.Ptr, OutputInfo.Size, y_gpu.Ptr, OutputInfo.Size, ActivationFunctions.Activation);
                    y_gpu.CopyToHost(z.Entities, z.Length, out a);
                }
            }
        }

        /// <inheritdoc/>
        public override unsafe void Backpropagate(in Tensor delta_1, in Tensor z, ActivationFunction activationPrime)
        {
            fixed (float* pw = Weights)
            {
                Tensor.Fix(pw, InputInfo.Size, OutputInfo.Size, out Tensor wSpan);
                using (DeviceMemory<float>
                    delta_1_gpu = DnnInstance.Gpu.AllocateDevice(delta_1),
                    w_gpu = DnnInstance.Gpu.AllocateDevice(wSpan),
                    z_gpu = DnnInstance.Gpu.AllocateDevice(z))
                {
                    DnnInstance.FullyConnectedBackwardData(z.Entities, InputInfo.Size, OutputInfo.Size, z_gpu.Ptr, z.Length, delta_1_gpu.Ptr, delta_1.Length, w_gpu.Ptr, wSpan.Length, activationPrime);
                    z_gpu.CopyTo(z);
                }
            }
        }

        /// <inheritdoc/>
        public override void ComputeGradient(in Tensor a, in Tensor delta, out Tensor dJdw, out Tensor dJdb)
        {
            Blas.TransposeAndMultiply(a, delta, out dJdw);
            delta.CompressVertically(out dJdb);
        }
    }
}
