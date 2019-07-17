using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Interfaces;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.Network.Nodes;
using NeuralNetworkDotNet.Network.Nodes.Unary.Abstract;

namespace NeuralNetworkDotNet.Network.Graph
{
    internal sealed class ComputationalGraph : APIs.Graph, INetwork
    {
        /// <inheritdoc/>
        public Shape InputShape => Input.Shape;

        /// <inheritdoc/>
        public Shape OutputShape => Output.Shape;

        /// <inheritdoc/>
        public int NodesCount => Nodes.Count;

        /// <inheritdoc/>
        public int ParametersCount => Nodes.OfType<WeightedUnaryNodeBase>().Sum(node => node.Parameters);

        /// <inheritdoc/>
        public bool IsInNumericOverflow => Nodes.OfType<WeightedUnaryNodeBase>().Any(node => node.IsInNumericOverflow);

        public ComputationalGraph([NotNull] IReadOnlyCollection<Node> nodes) : base(nodes) { }

        public bool Equals(INetwork other)
        {
            throw new NotImplementedException();
        }

        public INetwork Clone()
        {
            throw new NotImplementedException();
        }

        public Tensor Forward(Tensor x)
        {
            throw new NotImplementedException();
        }

        public float Loss(Tensor x, Tensor y)
        {
            throw new NotImplementedException();
        }

        public string SerializeMetadataAsJson()
        {
            throw new NotImplementedException();
        }

        public void Save(string path)
        {
            throw new NotImplementedException();
        }

        public void Save(Stream stream)
        {
            throw new NotImplementedException();
        }
    }
}
