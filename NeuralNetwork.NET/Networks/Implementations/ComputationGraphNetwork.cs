using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Graph;
using NeuralNetworkNET.Networks.Layers.Abstract;
using NeuralNetworkNET.SupervisedLearning.Data;
using NeuralNetworkNET.SupervisedLearning.Optimization;

namespace NeuralNetworkNET.Networks.Implementations
{
    /// <summary>
    /// A computational graph network with an arbitrary internal structure and number of computation branches
    /// </summary>
    internal sealed class ComputationGraphNetwork : NeuralNetworkBase
    {
        /// <inheritdoc/>
        public override ref readonly TensorInfo InputInfo => ref Graph.Root.Children[0].To<IComputationGraphNode, ProcessingNode>().Layer.InputInfo;
        
        /// <inheritdoc/>
        public override ref readonly TensorInfo OutputInfo => ref OutputLayer.OutputInfo;

        /// <inheritdoc/>
        public override IReadOnlyList<INetworkLayer> Layers { get; }

        /// <inheritdoc/>
        protected override OutputLayerBase OutputLayer { get; }

        /// <summary>
        /// The underlying layers graph for the network
        /// </summary>
        [NotNull]
        private readonly ComputationGraph Graph;

        public ComputationGraphNetwork([NotNull] IComputationGraphNode root) : base(NetworkType.ComputationGraph)
        {
            Graph = new ComputationGraph(root);
            OutputLayer = Graph.OutputNode.To<IComputationGraphNode, ProcessingNode>().Layer.To<INetworkLayer, OutputLayerBase>();
            Layers = Graph.Layers.Select(node => node.To<IComputationGraphNode, ProcessingNode>().Layer).ToArray();
        }

        public override IReadOnlyList<(float[] Z, float[] A)> ExtractDeepFeatures(float[] x)
        {
            throw new NotImplementedException();
        }

        public override IReadOnlyList<(float[,] Z, float[,] A)> ExtractDeepFeatures(float[,] x)
        {
            throw new NotImplementedException();
        }

        protected override void Forward(in Tensor x, out Tensor yHat)
        {
            throw new NotImplementedException();
        }

        internal override void Backpropagate(in SamplesBatch batch, float dropout, WeightsUpdater updater)
        {
            throw new NotImplementedException();
        }

        public override INeuralNetwork Clone()
        {
            throw new NotImplementedException();
        }
    }
}
