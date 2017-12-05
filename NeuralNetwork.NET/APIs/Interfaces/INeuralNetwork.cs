using System;
using System.Collections.Generic;
using System.IO;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers.Imaging;

namespace NeuralNetworkNET.APIs.Interfaces
{
    /// <summary>
    /// An interface to mask a neural network implementation
    /// </summary>
    public interface INeuralNetwork : IEquatable<INeuralNetwork>, IClonable<INeuralNetwork>
    {
        #region Properties

        /// <summary>
        /// Gets the size of the input layer
        /// </summary>
        int Inputs { get; }

        /// <summary>
        /// Gets the size of the output layer
        /// </summary>
        int Outputs { get; }

        /// <summary>
        /// Gets the list of layers in the network
        /// </summary>
        [NotNull, ItemNotNull]
        IReadOnlyList<INetworkLayer> Layers { get; }

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

        #endregion

        #region Tools

        /// <summary>
        /// Serializes the network as a JSON string
        /// </summary>
        [Pure, NotNull]
        String SerializeAsJSON();

        /// <summary>
        /// Saves the network in the target directory
        /// </summary>
        /// <param name="directory">The directory to use to save the network</param>
        /// <param name="name">The name for the network file to create</param>
        /// <returns>The path to the file that was created with the saved network</returns>
        [NotNull]
        String Save([NotNull] DirectoryInfo directory, [NotNull] String name);

        /// <summary>
        /// Exports the weights in the current network with a visual representation
        /// </summary>
        /// <param name="directory">The target directory</param>
        /// <param name="scaling">The desired image scaling to use</param>
        void ExportWeightsAsImages([NotNull] DirectoryInfo directory, ImageScaling scaling = ImageScaling.Native);

        #endregion
    }
}
