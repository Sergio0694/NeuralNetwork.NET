using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A class representing a split in a computation graph between an inference and a training branch
    /// </summary>
    public sealed class TrainingSplitNode : IComputationGraphNode
    {
        /// <inheritdoc/>
        public ComputationGraphNodeType Type { get; } = ComputationGraphNodeType.TrainingSplit;
    
        /// <summary>
        /// Gets the parent node for the current graph node
        /// </summary>
        [NotNull]
        public IComputationGraphNode Parent { get; }

        /// <summary>
        /// Gets the first node of the following inference branch
        /// </summary>
        [NotNull]
        public IComputationGraphNode InferenceBranchNode { get; }

        /// <summary>
        /// Gets the first node of the following training branch
        /// </summary>
        [NotNull]
        public IComputationGraphNode TrainingBranchNode { get; }

        private IReadOnlyList<IComputationGraphNode> _Children;

        /// <inheritdoc/>
        public IReadOnlyList<IComputationGraphNode> Children => _Children ?? (_Children = new[] { InferenceBranchNode, TrainingBranchNode });

        internal TrainingSplitNode([NotNull] IComputationGraphNode parent, [NotNull] IComputationGraphNode inference, [NotNull] IComputationGraphNode training)
        {
            Parent = parent;
            InferenceBranchNode = inference;
            TrainingBranchNode = training;
        }
    }
}
