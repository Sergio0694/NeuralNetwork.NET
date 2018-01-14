using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A graph of <see cref="INetworkLayer"/> instances, with O(1) pre-order access time for nodes
    /// </summary>
    public sealed class ComputationGraph : IReadOnlyList<INetworkLayer>
    {
        /// <summary>
        /// Gets the root <see cref="ComputationGraphNode"/> for the current graph
        /// </summary>
        [NotNull]
        public ComputationGraphNode Root { get; }

        // The in-order serialized view of the network layers in the graph
        [NotNull, ItemNotNull]
        private readonly IReadOnlyList<INetworkLayer> Layers;
        
        internal ComputationGraph(ComputationGraphNode root)
        {
            Root = root;

            // Explore the graph and build the serialized list
            List<INetworkLayer> layers = new List<INetworkLayer>();
            void Explore(ComputationGraphNode node)
            {
                layers.Add(node.Layer);
                foreach (ComputationGraphNode child in node.Children)
                    Explore(child);
            }
            Explore(root);
            Layers = layers;
        }

        #region Interface

        /// <inheritdoc/>
        public IEnumerator<INetworkLayer> GetEnumerator() => Layers.GetEnumerator();

        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <inheritdoc/>
        public int Count
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Layers.Count;
        }

        /// <inheritdoc/>
        public INetworkLayer this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Layers[index];
        }

        #endregion
    }
}
