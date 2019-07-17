using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkDotNet.Network.Nodes;
using NeuralNetworkDotNet.Network.Nodes.Nullary;
using NeuralNetworkDotNet.Network.Nodes.Unary.Losses;

namespace NeuralNetworkDotNet.APIs
{
    /// <summary>
    /// A <see langword="class"/> with primitives to build a computational graph
    /// </summary>
    public sealed class Graph
    {
        /// <summary>
        /// Gets the root <see cref="PlaceholderNode"/> for the current graph
        /// </summary>
        [NotNull]
        internal PlaceholderNode Input { get; }

        /// <summary>
        /// Gets the <see cref="OutputNode"/> instance for the current graph
        /// </summary>
        internal OutputNode Output { get; }

        /// <summary>
        /// Gets the collection of nodes for the current graph
        /// </summary>
        [NotNull, ItemNotNull]
        internal IReadOnlyCollection<Node> Nodes { get; }

        /// <summary>
        /// Creates a new <see cref="Graph"/> instance with the given collection of nodes
        /// </summary>
        /// <param name="nodes">The nodes to use to build the graph</param>
        internal Graph([NotNull] IReadOnlyCollection<Node> nodes)
        {
            Input = nodes.OfType<PlaceholderNode>().First();
            Output = nodes.OfType<OutputNode>().First();
            Nodes = nodes;
        }

        /// <summary>
        /// Creates a new <see cref="Node"/> instance for a linear input
        /// </summary>
        /// <param name="size">The input size</param>
        [Pure, NotNull]
        public static Node Linear(int size) => new PlaceholderNode((size, 1, 1));

        /// <summary>
        /// Creates a new <see cref="Node"/> instance for with a custom 3D shape
        /// </summary>
        /// <param name="height">The input volume height</param>
        /// <param name="width">The input volume width</param>
        /// <param name="channels">The number of channels in the input volume</param>
        [Pure, NotNull]
        public static Node Volume(int height, int width, int channels) => new PlaceholderNode((channels, height, width));
    }
}
