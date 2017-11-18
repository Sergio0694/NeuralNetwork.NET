using System;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Cost.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Helpers;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// An interface that represents a single layer in a multilayer neural network
    /// </summary>
    public interface INetworkLayer
    {
        /// <summary>
        /// Gets the number of inputs in the current layer
        /// </summary>
        int Inputs { get; }

        /// <summary>
        /// Gets the number of outputs in the current layer
        /// </summary>
        int Outputs { get; }
    }

    public abstract class NetworkLayerBase : INetworkLayer
    {
        /// <inheritdoc/>
        public abstract int Inputs { get; }

        /// <inheritdoc/>
        public abstract int Outputs { get; }

        /// <summary>
        /// Forwards the inputs through the network layer and returns the resulting activity (Z) and activation (A)
        /// </summary>
        /// <param name="x">The input to process</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public abstract (float[,] Z, float[,] A) Forward(float[,] x);

        /// <summary>
        /// Backpropagates the error to compute the delta for the inputs of the layer
        /// </summary>
        /// <param name="delta_1">The output error delta</param>
        /// <param name="z">The activity on the inputs of the layer</param>
        /// <param name="activationPrime">The activation prime function performed by the previous layer</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public abstract float[,] Backpropagate(float[,] delta_1, float[,] z, ActivationFunction activationPrime);
    }

    public abstract class WeightedLayerBase : NetworkLayerBase
    {
        /// <summary>
        /// Gets the weights for the current network layer
        /// </summary>
        [NotNull]
        public float[,] Weights { get; }

        /// <summary>
        /// Gets the biases for the current network layer
        /// </summary>
        [NotNull]
        public float[] Biases { get; }

        /// <summary>
        /// Gets the list of activation and activation prime functions used in the network
        /// </summary>
        public readonly (ActivationFunction Activation, ActivationFunction ActivationPrime) ActivationFunctions;

        protected WeightedLayerBase([NotNull] float[,] w, [NotNull] float[] b, ActivationFunctionType activation)
        {
            Weights = w;
            Biases = b;
            ActivationFunctions = ActivationFunctionProvider.GetActivations(activation);
        }
    }

    public class FullyConnectedLayer : WeightedLayerBase
    {
        /// <inheritdoc/>
        public override int Inputs => Weights.GetLength(0);

        /// <inheritdoc/>
        public override int Outputs => Weights.GetLength(1);

        public FullyConnectedLayer(int inputs, int outputs, ActivationFunctionType activation)
            : base(WeightsProvider.FullyConnectedWeights(inputs, outputs),
                  WeightsProvider.Biases(outputs), activation)
        { }

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
    }

    public abstract class OutputLayerBase : FullyConnectedLayer
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

        protected OutputLayerBase(int inputs, int outputs, ActivationFunctionType activation, CostFunctionType cost)
            : base(inputs, outputs, activation)
        {
            if (activation == ActivationFunctionType.Softmax && cost != CostFunctionType.LogLikelyhood ||
                cost == CostFunctionType.LogLikelyhood && activation != ActivationFunctionType.Softmax)
                throw new ArgumentException("The softmax activation and log-likelyhood cost function must be used together in a softmax layer");
            (CostFunction, CostFunctionPrime) = CostFunctionProvider.GetCostFunctions(cost);
        }

        /// <summary>
        /// Computes the output delta, with respect to the cost function of the network
        /// </summary>
        /// <param name="yHat">The estimated outputs for the network</param>
        /// <param name="y">The expected outputs for the used inputs</param>
        /// <param name="z">The activity on the output layer</param>
        public void Backpropagate(float[,] yHat, float[,] y, float[,] z)
        {
            CostFunctionPrime(yHat, y, z, ActivationFunctions.ActivationPrime);
        }
    }

    public class OutputLayer : OutputLayerBase
    {
        public OutputLayer(int inputs, int outputs, ActivationFunctionType activation, CostFunctionType cost)
            : base(inputs, outputs, activation, cost)
        {
            if (activation == ActivationFunctionType.Softmax)
                throw new ArgumentException("The softmax activation can only be used in a softmax layer");
        }
    }

    public sealed class SoftmaxLayer : OutputLayerBase
    {
        public SoftmaxLayer(int inputs, int outputs) 
            : base(inputs, outputs, ActivationFunctionType.Softmax, CostFunctionType.LogLikelyhood)
        { }

        /// <inheritdoc/>
        public override (float[,] Z, float[,] A) Forward(float[,] x)
        {
            (float[,] z, float[,] a) = base.Forward(x);
            a.InPlaceSoftmaxNormalization();
            return (z, a);
        }
    }

    public class ConvolutionLayer : WeightedLayerBase
    {
        public override int Inputs { get; }
        public override int Outputs { get; }

        public override (float[,] Z, float[,] A) Forward(float[,] x)
        {
            throw new NotImplementedException();
        }

        public override float[,] Backpropagate(float[,] delta_1, float[,] z, ActivationFunction activationPrime)
        {
            throw new NotImplementedException();
        }

        public ConvolutionLayer(int height, int width, int depth, int kernels, ActivationFunctionType activation) 
            : base(WeightsProvider.ConvolutionalKernels(height, width, depth, kernels), 
                  WeightsProvider.Biases(kernels), activation)
        { }
    }

    public class PoolingLayer : NetworkLayerBase
    {
        public override int Inputs { get; }
        public override int Outputs { get; }

        public PoolingLayer(int height, int width, int depth)
        {
            if (height <= 0 || width <= 0) throw new ArgumentOutOfRangeException("The height and width must be positive numbers");
            if (depth <= 0) throw new ArgumentOutOfRangeException(nameof(depth), "The depth must be at least equal to 1");
            Inputs = height * width * depth;
            Outputs = (height / 2 + (height % 2 == 0 ? 0 : 1)) * (width / 2 + (width % 2 == 0 ? 0 : 1)) * depth;
        }

        public override (float[,] Z, float[,] A) Forward(float[,] x)
        {
            throw new NotImplementedException();
        }

        public override float[,] Backpropagate(float[,] delta_1, float[,] z, ActivationFunction activationPrime)
        {
            throw new NotImplementedException();
        }
    }
}
