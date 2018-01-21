using System.Collections.Generic;
using System.Linq;
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
        /// Gets the child nodes for the following inference branch
        /// </summary>
        [NotNull, ItemNotNull]
        public IReadOnlyList<IComputationGraphNode> InferenceBranchNodes { get; }

        /// <summary>
        /// Gets the child nodes for the following training branch
        /// </summary>
        [NotNull, ItemNotNull]
        public IReadOnlyList<IComputationGraphNode> TrainingBranchNodes { get; }

        private IReadOnlyList<IComputationGraphNode> _Children;

        /// <inheritdoc/>
        public IReadOnlyList<IComputationGraphNode> Children => _Children ?? (_Children = InferenceBranchNodes.Concat(TrainingBranchNodes).ToArray());

        internal TrainingSplitNode([NotNull] IComputationGraphNode parent, [NotNull, ItemNotNull] IReadOnlyList<IComputationGraphNode> inference, [NotNull, ItemNotNull] IReadOnlyList<IComputationGraphNode> training)
        {
            Parent = parent;
            InferenceBranchNodes = inference;
            TrainingBranchNodes = training;
        }
    }
}
