using System;
using System.Collections.Generic;
using System.IO;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.APIs.Structs;

namespace NeuralNetworkNET.APIs.Interfaces
{
    /// <summary>
    /// An interface to mask a neural network implementation
    /// </summary>
    [PublicAPI]
    public interface INeuralNetwork : IEquatable<INeuralNetwork>, IClonable<INeuralNetwork>
    {
        #region Properties

        /// <summary>
        /// Gets the type of a given <see cref="INeuralNetwork"/> instance
        /// </summary>
        NetworkType NetworkType { get; }

        /// <summary>
        /// Gets the size of the input layer
        /// </summary>
        ref readonly TensorInfo InputInfo { get; }

        /// <summary>
        /// Gets the size of the output layer
        /// </summary>
        ref readonly TensorInfo OutputInfo { get; }

        /// <summary>
        /// Gets the list of layers in the network
        /// </summary>
        [NotNull, ItemNotNull]
        IReadOnlyList<INetworkLayer> Layers { get; }

        /// <summary>
        /// Gets the number of units in the current network (layers or graph nodes)
        /// </summary>
        int Size { get; }

        /// <summary>
        /// Gets the total number of parameters in the current network layer
        /// </summary>
        int Parameters { get; }

        /// <summary>
        /// Gets whether or not a numeric overflow has occurred in the network
        /// </summary>
        bool IsInNumericOverflow { get; }

        #endregion

        #region Methods

        /// <summary>
        /// Forwards the input through the network
        /// </summary>
        /// <param name="x">The input to process</param>
        /// <remarks>This methods processes a single input row and outputs a single result</remarks>
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        float[] Forward([NotNull] float[] x);

        /// <summary>
        /// Forwards the inputs through the network
        /// </summary>
        /// <param name="x">The input to process</param>
        /// <remarks>This methods processes multiple inputs at the same time, one per input row</remarks>
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        float[,] Forward([NotNull] float[,] x);

        /// <summary>
        /// Forwards the input through the network and returns a list of all the activity and activations computed by each layer
        /// </summary>
        /// <param name="x">The input to process</param>
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        IReadOnlyList<(float[] Z, float[] A)> ExtractDeepFeatures([NotNull] float[] x);

        /// <summary>
        /// Forwards the inputs through the network and returns a list of all the activity and activations computed by each layer
        /// </summary>
        /// <param name="x">The input to process</param>
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        IReadOnlyList<(float[,] Z, float[,] A)> ExtractDeepFeatures([NotNull] float[,] x);

        /// <summary>
        /// Calculates the cost function for the current instance and the input values
        /// </summary>
        /// <param name="x">The input values for the network</param>
        /// <param name="y">The expected result to use to calculate the error</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        float CalculateCost([NotNull] float[] x, [NotNull] float[] y);

        /// <summary>
        /// Calculates the cost function for the current instance and the input values
        /// </summary>
        /// <param name="x">The input values for the network</param>
        /// <param name="y">The expected result to use to calculate the error</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        float CalculateCost([NotNull] float[,] x, [NotNull] float[,] y);

        /// <summary>
        /// Calculates the current network performances with the given dataset
        /// </summary>
        /// <param name="dataset">The <see cref="IDataset"/> instance to evaluate</param>
        (float Cost, int Classified, float Accuracy) Evaluate([NotNull] IDataset dataset);

        #endregion

        #region Serialization

        /// <summary>
        /// Serializes the network metadata as a JSON string
        /// </summary>
        [Pure, NotNull]
        string SerializeMetadataAsJson();

        /// <summary>
        /// Saves the network to the target file
        /// </summary>
        /// <param name="target">The <see cref="FileInfo"/> instance for the target file (it may not exist yet)</param>
        void Save([NotNull] FileInfo target);

        /// <summary>
        /// Saves the network to the target stream
        /// </summary>
        /// <param name="stream">The <see cref="Stream"/> to use to write the network data</param>
        void Save([NotNull] Stream stream);

        #endregion
    }
}
