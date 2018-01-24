using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Graph.Nodes.Abstract;

namespace NeuralNetworkNET.Networks.Graph.Nodes
{
    /// <summary>
    /// A class representing a sum node in a computation graph
    /// </summary>
    internal abstract class SumNode : MergeNodeBase
    {
        #region Initialization

        /// <summary>
        /// Gets the activation type used in the current node
        /// </summary>
        public ActivationFunctionType ActivationFunctionType { get; }

        /// <summary>
        /// Gets the list of activation and activation prime functions used in the sum node
        /// </summary>
        public (ActivationFunction Activation, ActivationFunction ActivationPrime) ActivationFunctions { get; }

        protected SumNode(ActivationFunctionType activation, [NotNull] [ItemNotNull] IReadOnlyList<IComputationGraphNode> parents) : base(ComputationGraphNodeType.Sum, parents)
        {
            ActivationFunctionType = activation;
            ActivationFunctions = ActivationFunctionProvider.GetActivations(activation);
        }

        /// <summary>
        /// Creates a new <see cref="SumNode"/> with the given parameters
        /// </summary>
        /// <param name="mode">The desired execution mode</param>
        /// <param name="activation">The sum node activation function</param>
        /// <param name="parents">The parent nodes for the new sum mode to create</param>
        [Pure, NotNull]
        public static SumNode New(ExecutionModePreference mode, ActivationFunctionType activation, [NotNull] [ItemNotNull] IReadOnlyList<IComputationGraphNode> parents)
        {
            if (mode == ExecutionModePreference.Cpu) return new CpuSumNode(activation, parents);
            return new CudaSumNode(activation, parents);
        }

        #endregion

        /// <summary>
        /// A CPU-powered sum node
        /// </summary>
        private sealed class CpuSumNode : SumNode
        {
            public CpuSumNode(ActivationFunctionType activation, [NotNull] [ItemNotNull] IReadOnlyList<IComputationGraphNode> parents) 
                : base(activation, parents) { }
        }

        /// <summary>
        /// A CUDA-powered sum node
        /// </summary>
        private sealed class CudaSumNode : SumNode
        {
            public CudaSumNode(ActivationFunctionType activation, [NotNull] [ItemNotNull] IReadOnlyList<IComputationGraphNode> parents) 
                : base(activation, parents) { }
        }
    }
}
