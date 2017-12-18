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
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            using (DeviceMemory2D<float>
                x_gpu = DnnInstance.Gpu.AllocateDevice2D(x),
                w_gpu = DnnInstance.Gpu.AllocateDevice(Weights),
                y_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities, OutputInfo.Size))
            using (DeviceMemory<float> b_gpu = DnnInstance.Gpu.AllocateDevice(Biases))
            {
                DnnInstance.FullyConnectedForward(x.Entities, x.Length, OutputInfo.Size, x_gpu.Ptr, x_gpu.PitchInElements.ToInt32(), w_gpu.Ptr, w_gpu.PitchInElements.ToInt32(), b_gpu.Ptr, y_gpu.Ptr, y_gpu.PitchInElements.ToInt32());
                y_gpu.CopyToHost(out z);
                DnnInstance.ActivationForward(z.Entities, z.Length, y_gpu.Ptr, y_gpu.PitchInElements.ToInt32(), y_gpu.Ptr, y_gpu.PitchInElements.ToInt32(), ActivationFunctions.Activation);
                y_gpu.CopyToHost(out a);
            }
        }

        /// <inheritdoc/>
        public override void Backpropagate(in Tensor delta_1, in Tensor z, ActivationFunction activationPrime)
        {
            using (DeviceMemory2D<float>
                delta_1_gpu = DnnInstance.Gpu.AllocateDevice2D(delta_1),
                w_gpu = DnnInstance.Gpu.AllocateDevice(Weights),
                z_gpu = DnnInstance.Gpu.AllocateDevice2D(z))
            {
                DnnInstance.FullyConnectedBackwardData(z.Entities, InputInfo.Size, OutputInfo.Size, z_gpu.Ptr, z_gpu.PitchInElements.ToInt32(), delta_1_gpu.Ptr, delta_1_gpu.PitchInElements.ToInt32(), w_gpu.Ptr, w_gpu.PitchInElements.ToInt32(), activationPrime);
                z_gpu.CopyTo(z);
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
