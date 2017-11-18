using System;
using System.Collections.Generic;
using System.Text;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Cost.Delegates;
using NeuralNetworkNET.Networks.Implementations.Misc;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    public interface INetworkLayer
    {
        int Inputs { get; }

        int Outputs { get; }

        (float[,] Z, float[,] A) Forward(float[,] x);
    }

    public abstract class NetworkLayerBase : INetworkLayer
    {
        public abstract int Inputs { get; }

        public abstract int Outputs { get; }

        public abstract (float[,] Z, float[,] A) Forward(float[,] x);

        public abstract void ValidateInputSize(int size);

        public abstract float[,] Backpropagate(float[,] delta_1, float[,] z, ActivationFunction activationPrime);
    }

    public abstract class WeightedLayerBase : NetworkLayerBase
    {
        public float[,] Weights { get; }

        public float[] Biases { get; }

        /// <summary>
        /// Gets the list of activation and activation prime functions used in the network
        /// </summary>
        public readonly (ActivationFunction Activation, ActivationFunction ActivationPrime) ActivationFunctions;
    }

    public class FullyConnectedLayer : WeightedLayerBase
    {
        public override int Inputs { get; }
        public override int Outputs { get; }

        public override (float[,] Z, float[,] A) Forward(float[,] x)
        {
            float[,]
                z = MatrixServiceProvider.MultiplyWithSum(x, Weights, Biases),
                a = MatrixServiceProvider.Activation(z, ActivationFunctions.Activation);
            return (z, a);
        }

        public override void ValidateInputSize(int size)
        {
            if (size != Inputs) throw new ArgumentOutOfRangeException(nameof(Inputs), "The number of inputs from the previous layer isn't valid");
        }

        public override float[,] Backpropagate(float[,] delta_1, float[,] z, ActivationFunction activationPrime)
        {
            float[,] wt = Weights.Transpose();
            MatrixServiceProvider.InPlaceMultiplyAndHadamardProductWithAcrivationPrime(z, delta_1, wt, activationPrime);
            return z;
        }
    }

    public class OutputLayer : FullyConnectedLayer
    {
        /// <summary>
        /// Gets the <see cref="CostFunction"/> used to evaluate the neural network
        /// </summary>
        [NotNull]
        private readonly CostFunction CostFunction;

        /// <summary>
        /// Gets the <see cref="CostFunctionPrime"/> used in the gradient descent algorithm
        /// </summary>
        [NotNull]
        private readonly CostFunctionPrime CostFunctionPrime;

        public void Backpropagate(float[,] yHat, float[,] y, float[,] z)
        {
            CostFunctionPrime(yHat, y, z, ActivationFunctions.ActivationPrime);
        }
    }

    public sealed class SoftmaxLayer : OutputLayer
    {
        public override (float[,] Z, float[,] A) Forward(float[,] x)
        {
            (float[,] z, float[,] a) = base.Forward(x);
            a.InPlaceSoftmaxNormalization();
            return (z, a);
        }
    }
}
