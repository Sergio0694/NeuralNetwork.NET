using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Exceptions;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Layers.Abstract;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A graph of <see cref="INetworkLayer"/> instances, with O(1) pre-order access time for nodes
    /// </summary>
    internal sealed class ComputationGraph : IEquatable<ComputationGraph>
    {
        #region Parameters

        /// <summary>
        /// Gets the root <see cref="InputNode"/> for the current graph
        /// </summary>
        [NotNull]
        public InputNode Root { get; }

        /// <summary>
        /// Gets the in-order serialized view of the network graph nodes
        /// </summary>
        [NotNull, ItemNotNull]
        internal readonly IReadOnlyList<IComputationGraphNode> Nodes;

        /// <summary>
        /// Gets the in-order serialized view of the <see cref="ProcessingNode"/> instances in the current graph
        /// </summary>
        [NotNull, ItemNotNull]
        internal readonly IReadOnlyList<ProcessingNode> ProcessingNodes;

        /// <summary>
        /// Gets the graph main output node
        /// </summary>
        [NotNull]
        internal readonly ProcessingNode OutputNode;

        /// <summary>
        /// Gets the training output nodes, if present
        /// </summary>
        [NotNull, ItemNotNull]
        internal readonly IReadOnlyList<ProcessingNode> TrainingOutputNodes;

        #endregion
        
        #region Initialization

        private ComputationGraph(IComputationGraphNode root, IReadOnlyList<IComputationGraphNode> nodes, ProcessingNode output, IReadOnlyList<ProcessingNode> trainingOutputs)
        {
            Root = root is InputNode input ? input : throw new ArgumentException("The root node isn't valid");
            Nodes = nodes;
            OutputNode = output;
            TrainingOutputNodes = trainingOutputs;
            ProcessingNodes = Nodes.Pick<IComputationGraphNode, ProcessingNode>().ToArray();
        }

        /// <summary>
        /// Builds a computation graph from the input node and <see cref="TensorInfo"/> shape
        /// </summary>
        /// <param name="input">The shape of the inputs for the graph to create</param>
        /// <param name="root">The <see cref="NodeBuilder"/> instance that represents the graph root node</param>
        [Pure, NotNull]
        public static ComputationGraph New(TensorInfo input, [NotNull] NodeBuilder root)
        {
            // Captured dictionary to keep track of graph nodes and tensor shape for each builder
            Dictionary<NodeBuilder, (NodeBase Node, TensorInfo Info, Guid Id)> map = new Dictionary<NodeBuilder, (NodeBase, TensorInfo, Guid)>();
            List<IComputationGraphNode> nodes = new List<IComputationGraphNode>();
            ProcessingNode output = null;
            List<ProcessingNode> trainingOutputs = new List<ProcessingNode>();

            // Function to build the computation graph with a top-down direction
            void BuildMap(NodeBuilder node, Guid id)
            {
                // Node check
                if (map.TryGetValue(node, out var value))
                {
                    if (value.Id != id) throw new ComputationGraphBuildException("The training branch can't cross other graph branches");
                    return;
                }

                // Process the current node
                NodeBase iNode;
                switch (node.NodeType)
                {
                    case ComputationGraphNodeType.Input:
                        if (node.Parents.Count > 0) throw new ComputationGraphBuildException("An input node can't haave any parent nodes");
                        if (node.Children.Count == 0) throw new ComputationGraphBuildException("An input node can't have 0 child nodes");
                        iNode = new InputNode();
                        map[node] = (iNode, input, id);
                        break;
                    case ComputationGraphNodeType.Processing:
                        if (node.Factory == null) throw new ComputationGraphBuildException("Missing layer factory");
                        INetworkLayer layer = node.Factory(map[node.Parents[0]].Info);
                        ProcessingNode processing = new ProcessingNode(layer, map[node.Parents[0]].Node);
                        if (layer is OutputLayerBase)
                        {
                            if (output != null) throw new ComputationGraphBuildException("The graph can only have a single inference output node");
                            if (node.Children.Count > 0) throw new ComputationGraphBuildException("An output node can't have any child nodes");
                            if (id != default)
                            {
                                if (map.Values.Any(entry => entry.Node is ProcessingNode p && p.Layer is OutputLayerBase && entry.Id == id))
                                    throw new ComputationGraphBuildException("Each training branch can have a single output node");
                                trainingOutputs.Add(processing);
                            }
                            output = processing;
                        }
                        else if (node.Children.Count == 0) throw new ComputationGraphBuildException("A processing node can't have 0 child nodes");
                        iNode = processing;
                        map[node] = (iNode, layer.OutputInfo, id);
                        break;
                    case ComputationGraphNodeType.TrainingBranch:
                        if (id != default) throw new ComputationGraphBuildException("A training branch can't contain secondary training branches");
                        if (node.Children.Count == 0) throw new ComputationGraphBuildException("A training branch node can't have 0 child nodes");
                        iNode = new TrainingNode(map[node.Parents[0]].Node);
                        map[node] = (iNode, map[node.Parents[0]].Info, id);
                        id = Guid.NewGuid();
                        break;
                    case ComputationGraphNodeType.DepthConcatenation:
                    case ComputationGraphNodeType.Sum:
                        if (node.Children.Count == 0) throw new ComputationGraphBuildException("A merge node can't have 0 child nodes");

                        // Gather the parent nodes
                        var parents = new (NodeBase Node, TensorInfo Info, Guid)[node.Parents.Count];
                        foreach ((NodeBuilder parent, int i) in node.Parents.Select((p, i) => (p, i)))
                            if (!map.TryGetValue(parent, out parents[i]))
                                return;

                        // Calculate the output tensor size
                        TensorInfo shape = node.NodeType == ComputationGraphNodeType.Sum 
                            ? parents[0].Info 
                            : TensorInfo.Volume(parents[0].Info.Height, parents[0].Info.Width, parents.Sum(p => p.Info.Channels));
                        iNode = new MergeNode(node.NodeType, parents.Select(t => t.Node).ToArray());
                        map[node] = (iNode, shape, id);
                        break;
                    default:
                        throw new ComputationGraphBuildException($"Invalid node type: {node.NodeType}");
                }
                nodes.Add(iNode);
                foreach (NodeBuilder child in node.Children)
                    BuildMap(child, id);
            }

            // Build the graph and bind the child nodes
            BuildMap(root, default);
            if (output == null) throw new ComputationGraphBuildException("The input graph doesn't have a valid output node");
            if (trainingOutputs.Any(node => node.Layer.OutputInfo != output.Layer.OutputInfo)) throw new ComputationGraphBuildException("The training outputs must match the main output tensor shape");
            foreach (KeyValuePair<NodeBuilder, (NodeBase Node, TensorInfo Info, Guid _)> pair in map)
            {
                pair.Value.Node.Children = pair.Key.Children.Select(child => map[child].Node).ToArray();
            }
            return new ComputationGraph(map[root].Node, nodes, output, trainingOutputs);
        }

        #endregion

        #region Tools

        /// <inheritdoc/>
        public bool Equals(ComputationGraph other)
        {
            // Setup
            if (other == null) return false;
            if (other == this) return true;
            if (Nodes.Count != other.Nodes.Count) return false;

            // Function to extract the indexes of the child nodes for a target node
            int[] GetIndexes(IEnumerable<IComputationGraphNode> nodes, IReadOnlyList<IComputationGraphNode> list)
            {
                List<int> indexes = new List<int>();
                foreach (IComputationGraphNode node in nodes)
                    for (int i = 0; i < list.Count; i++)
                        if (list[i] == node)
                        {
                            indexes.Add(i);
                            break;
                        }
                return indexes.ToArray();
            }

            // Perform the actual comparison
            for (int i = 0; i < Nodes.Count; i++)
            {
                IComputationGraphNode n1 = Nodes[i], n2 = other.Nodes[i];
                if (!n1.Equals(n2) ||
                    !GetIndexes(n1.Children, Nodes).SequenceEqual(GetIndexes(n2.Children, other.Nodes))) return false;
                switch (n1)
                {
                    case MergeNode merge:
                        if (!GetIndexes(merge.Parents, Nodes).SequenceEqual(GetIndexes(n2.To<IComputationGraphNode, MergeNode>().Parents, other.Nodes))) return false;
                        break;
                    case ProcessingNode processing:
                        if (Nodes.IndexOf(processing.Parent) != other.Nodes.IndexOf(n2.To<IComputationGraphNode, ProcessingNode>().Parent)) return false;
                        break;
                    case TrainingNode split:
                        if (Nodes.IndexOf(split.Parent) != other.Nodes.IndexOf(n2.To<IComputationGraphNode, ProcessingNode>().Parent)) return false;
                        break;
                    case InputNode _: break;
                    default: throw new InvalidOperationException("The graph contains an invalid node");
                }
            }
            return true;
        }

        /// <summary>
        /// Writes the current graph to the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The target <see cref="Stream"/> to use to write the graph data</param>
        public void Serialize([NotNull] Stream stream)
        {
            // Write the graph nodes
            stream.Write(Nodes.Count);
            Dictionary<IComputationGraphNode, Guid> map = Nodes.ToDictionary(n => n, _ => Guid.NewGuid());
            foreach (NodeBase node in Nodes.Cast<NodeBase>())
            {
                stream.Write(map[node]);
                node.Serialize(stream);
            }

            // Write the neighbours of each node
            foreach (IComputationGraphNode node in Nodes)
            {
                stream.Write(map[node]);
                stream.Write(node.Children.Count);
                foreach (IComputationGraphNode child in node.Children)
                    stream.Write(map[child]);
                switch (node)
                {
                    case ProcessingNode processing:
                        stream.Write(1);
                        stream.Write(map[processing.Parent]);
                        break;
                    case MergeNode merge:
                        stream.Write(merge.Parents.Count);
                        foreach (IComputationGraphNode parent in merge.Parents)
                        stream.Write(map[parent]);
                        break;
                    case TrainingNode split:
                        stream.Write(1);
                        stream.Write(map[split.Parent]);
                        break;
                    case InputNode _: break;
                    default: throw new InvalidOperationException("Invalid graph node type");
                }
            }
        }

        #endregion
    }
}
