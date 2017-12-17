using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.Cuda.Extensions;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers;
using NeuralNetworkNET.Structs;
namespace NeuralNetworkNET.Cuda.Layers
{
    internal class CuDnnFullyConnectedLayer : FullyConnectedLayer
    {
        public CuDnnFullyConnectedLayer(int inputs, int outputs, ActivationFunctionType activation) 
            : base(inputs, outputs, activation) { }

        public CuDnnFullyConnectedLayer([NotNull] float[,] weights, [NotNull] float[] biases, ActivationFunctionType activation) 
            : base(weights, biases, activation) { }

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<float>
                x_gpu = gpu.AllocateDevice2D(x),
                w_gpu = gpu.AllocateDevice(Weights),
                y_gpu = gpu.AllocateDevice<float>(x.Entities, Outputs))
            using (DeviceMemory<float> b_gpu = gpu.AllocateDevice(Biases))
            {
                Dnn dnn = Dnn.Get(gpu);
                dnn.FullyConnectedForward(x.Entities, x.Length, Outputs, x_gpu.Ptr, x_gpu.PitchInElements.ToInt32(), w_gpu.Ptr, w_gpu.PitchInElements.ToInt32(), b_gpu.Ptr, y_gpu.Ptr, y_gpu.PitchInElements.ToInt32());
                y_gpu.CopyToHost(out z);
                dnn.ActivationForward(z.Entities, z.Length, y_gpu.Ptr, y_gpu.PitchInElements.ToInt32(), y_gpu.Ptr, y_gpu.PitchInElements.ToInt32(), ActivationFunctions.Activation);
                y_gpu.CopyToHost(out a);
            }
        }

        /// <inheritdoc/>
        public override void Backpropagate(in Tensor delta_1, in Tensor z, ActivationFunction activationPrime)
        {
            Weights.Transpose(out Tensor wt);
            Blas.InPlaceMultiplyAndHadamardProductWithActivationPrime(z, delta_1, wt, activationPrime);
            wt.Free();
        }

        /// <inheritdoc/>
        public override void ComputeGradient(in Tensor a, in Tensor delta, out Tensor dJdw, out Tensor dJdb)
        {
            Blas.TransposeAndMultiply(a, delta, out dJdw);
            delta.CompressVertically(out dJdb);
        }
    }
}
