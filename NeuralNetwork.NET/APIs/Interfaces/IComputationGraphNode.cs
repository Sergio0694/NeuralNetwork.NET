using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;

namespace NeuralNetworkNET.APIs.Interfaces
{
    /// <summary>
    /// The base <see langword="interface"/> for the various types of nodes in a graph network
    /// </summary>
    public interface IComputationGraphNode : IEquatable<IComputationGraphNode>
    {
        /// <summary>
        /// Indicates the type of the current node
        /// </summary>
        ComputationGraphNodeType Type { get; }

        /// <summary>
        /// Gets the collection of child nodes for the current computation graph node
        /// </summary>
        [NotNull, ItemNotNull]
        IReadOnlyList<IComputationGraphNode> Children { get; }
    }
}
