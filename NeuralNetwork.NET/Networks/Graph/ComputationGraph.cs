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
        /// Gets the root <see cref="ProcessingNode"/> for the current graph
        /// </summary>
        [NotNull]
        public ProcessingNode Root { get; }

        // The in-order serialized view of the network layers in the graph
        [NotNull, ItemNotNull]
        private readonly IReadOnlyList<INetworkLayer> Layers;
        
        internal ComputationGraph(ProcessingNode root)
        {
            Root = root;
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
