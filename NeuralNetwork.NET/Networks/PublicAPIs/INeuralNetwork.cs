using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Activations;

namespace NeuralNetworkNET.Networks.PublicAPIs
{
    /// <summary>
    /// An interface to mask a neural network implementation
    /// </summary>
    public interface INeuralNetwork : IEquatable<INeuralNetwork>
    {
        /// <summary>
        /// Gets the size of the input layer
        /// </summary>
        int InputLayerSize { get; }

        /// <summary>
        /// Gets the size of the output layer
        /// </summary>
        int OutputLayerSize { get; }

        /// <summary>
        /// Gets the description of the network hidden layers
        /// </summary>
        [NotNull]
        IReadOnlyList<int> HiddenLayers { get; }

        /// <summary>
        /// Gets the list of activation functions used in the network layers
        /// </summary>
        [NotNull]
        IReadOnlyList<ActivationFunctionType> ActivationFunctions { get; }

        /// <summary>
        /// Forwards the input through the network
        /// </summary>
        /// <param name="input">The input to process</param>
        /// <remarks>This methods processes a single input row and outputs a single result</remarks>
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        float[] Forward([NotNull] float[] input);

        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        float[,] Forward([NotNull] float[,] x);

        /// <summary>
        /// Calculates the cost function for the current instance and the input values
        /// </summary>
        /// <param name="input">The input values for the network</param>
        /// <param name="y">The expected result to use to calculate the error</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        float CalculateCost([NotNull] float[] input, [NotNull] float[] y);

        /// <summary>
        /// Calculates the cost function for the current instance and the input values
        /// </summary>
        /// <param name="input">The input values for the network</param>
        /// <param name="y">The expected result to use to calculate the error</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        float CalculateCost([NotNull] float[,] input, [NotNull] float[,] y);

        /// <summary>
        /// Serializes the network as a JSON string
        /// </summary>
        [Pure, NotNull]
        String SerializeAsJSON();
    }
}
