using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using NeuralNetworkNET.Networks.Implementations.Layers.Helpers;
using NeuralNetworkNET.Networks.Implementations.Misc;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// A fully connected (dense) network layer
    /// </summary>
    internal class FullyConnectedLayer : WeightedLayerBase
    {
        /// <inheritdoc/>
        public override int Inputs => Weights.GetLength(0);

        /// <inheritdoc/>
        public override int Outputs => Weights.GetLength(1);

        public FullyConnectedLayer(int inputs, int outputs, ActivationFunctionType activation)
            : base(WeightsProvider.FullyConnectedWeights(inputs, outputs),
                WeightsProvider.Biases(outputs), activation)
        { }

        protected FullyConnectedLayer([NotNull] float[,] weights, [NotNull] float[] biases, ActivationFunctionType activation)
            : base(weights, biases, activation) { }

        /// <inheritdoc/>
        public override (float[,] Z, float[,] A) Forward(float[,] x)
        {
            float[,]
                z = MatrixServiceProvider.MultiplyWithSum(x, Weights, Biases),
                a = MatrixServiceProvider.Activation(z, ActivationFunctions.Activation);
            return (z, a);
        }

        /// <inheritdoc/>
        public override float[,] Backpropagate(float[,] delta_1, float[,] z, ActivationFunction activationPrime)
        {
            float[,] wt = Weights.Transpose();
            MatrixServiceProvider.InPlaceMultiplyAndHadamardProductWithAcrivationPrime(z, delta_1, wt, activationPrime);
            return z;
        }

        /// <inheritdoc/>
        public override LayerGradient ComputeGradient(float[,] a, float[,] delta)
        {
            // Compute dJdw(l) and dJdb(l)
            float[,] dJdw = MatrixServiceProvider.TransposeAndMultiply(a, delta); // dJdWi, previous activation transposed * current delta
            float[] dJdb = delta.CompressVertically();
            return new LayerGradient(dJdw, dJdb);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new FullyConnectedLayer(Weights.BlockCopy(), Biases.BlockCopy(), ActivationFunctionType);
    }
}
