using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Interfaces;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.Extensions.Types;
using NeuralNetworkDotNet.Network.Nodes;
using NeuralNetworkDotNet.Network.Nodes.Abstract;
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

        /// <inheritdoc/>
        public Tensor Forward(Tensor x)
        {
            using (var map = new DisposableDictionary<Node, Tensor> { [Input] = x })
            {
                // Recursive function to forward the inputs through the graph
                void Forward(Node node)
                {
                    for (var i = 0; i < node.Children.Count; i++)
                    {
                        var child = node.Children[i];
                        switch (child)
                        {
                            case UnaryNodeBase unary:
                                map[unary] = unary.Forward(map[node]);
                                break;
                            case BinaryNodeBase binary:
                                if (map.TryGetValue(binary.LeftParent, out var x1) &&
                                    map.TryGetValue(binary.RightParent, out var x2))
                                {
                                    map[binary] = binary.Forward(x1, x2);
                                }
                                break;
                            default: throw new InvalidOperationException("Invalid network graph");
                        }

                        Forward(child);
                    }
                }

                // Forward the input tensor starting from the input node
                Forward(Input);

                // Get the output tensor and remove it from the mapping to keep it alive
                var yHat = map[Output];
                map.Remove(Output);

                return yHat;
            }
        }

        /// <inheritdoc/>
        public float Loss(Tensor x, Tensor y)
        {
            using (var yHat = Forward(x))
            {
                return Output.CostFunctions.Cost(yHat, y);
            }
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
