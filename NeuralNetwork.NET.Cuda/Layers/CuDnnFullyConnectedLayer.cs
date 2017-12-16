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
            Blas.MultiplyWithSum(x, Weights, Biases, out z);
            Blas.Activation(z, ActivationFunctions.Activation, out a);
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
