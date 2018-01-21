using System;
using System.Collections.Generic;
using System.Text;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A class that represents the root node for a training sub-graph
    /// </summary>
    internal sealed class TrainingNode : IComputationGraphNode
    {
        /// <inheritdoc/>
        public ComputationGraphNodeType Type { get; } = ComputationGraphNodeType.TrainingBranch;

        /// <summary>
        /// Gets the root node for the current sub-graph
        /// </summary>
        [NotNull]
        public IComputationGraphNode Parent { get; }

        /// <inheritdoc/>
        public IReadOnlyList<IComputationGraphNode> Children { get; }

        public TrainingNode([NotNull] IComputationGraphNode root, [NotNull, ItemNotNull] IReadOnlyList<IComputationGraphNode> children)
        {
            Parent = root;
            Children = children;
        }
    }
}
