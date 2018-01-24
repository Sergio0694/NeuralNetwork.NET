using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.Networks.Graph.Nodes.Abstract;

namespace NeuralNetworkNET.Networks.Graph.Nodes
{
    /// <summary>
    /// A class representing a depth concatenation node
    /// </summary>
    internal sealed class DepthConcatenationNode : MergeNodeBase
    {
        internal DepthConcatenationNode([NotNull, ItemNotNull] IReadOnlyList<IComputationGraphNode> parents) : base(ComputationGraphNodeType.DepthConcatenation, parents) { }
    }
}
