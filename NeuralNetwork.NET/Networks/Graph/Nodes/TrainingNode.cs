using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.Networks.Graph.Nodes.Abstract;

namespace NeuralNetworkNET.Networks.Graph.Nodes
{
    /// <summary>
    /// A class that represents the root node for a training sub-graph
    /// </summary>
    internal sealed class TrainingNode : NodeBase
    {
        /// <summary>
        /// Gets the root node for the current sub-graph
        /// </summary>
        [NotNull]
        public IComputationGraphNode Parent { get; }

        public TrainingNode([NotNull] IComputationGraphNode root) : base(ComputationGraphNodeType.TrainingBranch) => Parent = root;
    }
}
