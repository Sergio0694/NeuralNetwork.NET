using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.Networks.Graph.Nodes.Abstract;

namespace NeuralNetworkNET.Networks.Graph.Nodes
{
    /// <summary>
    /// A class representing a sum node in a computation graph
    /// </summary>
    internal sealed class SumNode : MergeNodeBase
    {
        public SumNode([NotNull] [ItemNotNull] IReadOnlyList<IComputationGraphNode> parents) : base(ComputationGraphNodeType.Sum, parents) { }
    }
}
