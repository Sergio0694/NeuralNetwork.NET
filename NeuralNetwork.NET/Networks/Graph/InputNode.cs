using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A class that represents the root node for a computation graph
    /// </summary>
    internal sealed class InputNode : IComputationGraphNode
    {
        /// <inheritdoc/>
        public ComputationGraphNodeType Type { get; } = ComputationGraphNodeType.Input;

        /// <inheritdoc/>
        public IReadOnlyList<IComputationGraphNode> Children { get; }

        public InputNode([NotNull, ItemNotNull] IReadOnlyList<IComputationGraphNode> children) => Children = children;
    }
}
