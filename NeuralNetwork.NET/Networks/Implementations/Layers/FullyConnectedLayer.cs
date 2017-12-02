using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using NeuralNetworkNET.Networks.Implementations.Layers.Helpers;
using NeuralNetworkNET.Structs;

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

        public FullyConnectedLayer([NotNull] float[,] weights, [NotNull] float[] biases, ActivationFunctionType activation)
            : base(weights, biases, activation) { }

        /// <inheritdoc/>
        public override void Forward(in FloatSpan2D x, out FloatSpan2D z, out FloatSpan2D a)
        {
            MatrixServiceProvider.MultiplyWithSum(x, Weights, Biases, out z);
            MatrixServiceProvider.Activation(z, ActivationFunctions.Activation, out a);
        }

        /// <inheritdoc/>
        public override void Backpropagate(in FloatSpan2D delta_1, in FloatSpan2D z, ActivationFunction activationPrime)
        {
            Weights.Transpose(out FloatSpan2D wt);
            MatrixServiceProvider.InPlaceMultiplyAndHadamardProductWithActivationPrime(z, delta_1, wt, activationPrime);
        }

        /// <inheritdoc/>
        public override void ComputeGradient(in FloatSpan2D a, in FloatSpan2D delta, out FloatSpan2D dJdw, out FloatSpan dJdb)
        {
            MatrixServiceProvider.TransposeAndMultiply(a, delta, out dJdw);
            dJdw.CompressVertically(out dJdb);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new FullyConnectedLayer(Weights.BlockCopy(), Biases.BlockCopy(), ActivationFunctionType);
    }
}
