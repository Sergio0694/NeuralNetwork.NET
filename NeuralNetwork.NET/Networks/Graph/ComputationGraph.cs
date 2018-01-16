using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A graph of <see cref="INetworkLayer"/> instances, with O(1) pre-order access time for nodes
    /// </summary>
    public sealed class ComputationGraph
    {
        /// <summary>
        /// Gets the root <see cref="IComputationGraphNode"/> for the current graph
        /// </summary>
        [NotNull]
        public IComputationGraphNode Root { get; }

        /// <summary>
        /// Gets the in-order serialized view of the network graph nodes
        /// </summary>
        [NotNull, ItemNotNull]
        internal readonly IReadOnlyList<IComputationGraphNode> Layers;
        
        internal ComputationGraph(IComputationGraphNode root)
        {
            Root = root;
            Layers = TrySerializeLayers(root) ?? throw new ArgumentException("The input graph isn't valid", nameof(root));
        }

        #region Tools

        /// <summary>
        /// Tries to serialize the input graph into a sequence of <see cref="IComputationGraphNode"/> instances, if possible
        /// </summary>
        /// <param name="root">The root <see cref="IComputationGraphNode"/> for the computation graph</param>
        [Pure, CanBeNull]
        private static IReadOnlyList<IComputationGraphNode> TrySerializeLayers(IComputationGraphNode root)
        {
            HashSet<IComputationGraphNode> nodes = new HashSet<IComputationGraphNode>();

            bool Explore(IComputationGraphNode node)
            {
                // Add the current node, if not existing
                if (nodes.Contains(node)) return false;
                nodes.Add(node);

                // Explore the graph
                switch (node)
                {
                    case ProcessingNode processing:
                        if (!processing.Children.All(Explore)) return false;
                        break;
                    case MergeNode merge:
                        if (!merge.Children.All(Explore)) return false;
                        break;
                    case TrainingSplitNode split:
                        if (!Explore(split.InferenceBranchNode)) return false;
                        if (!Explore(split.TrainingBranchNode)) return false;
                        break;
                    default: throw new ArgumentException("Invalid node type", nameof(node));
                }
                return true;
            }

            return Explore(root) ? nodes.ToArray() : null;
        }

        #endregion
    }
}
