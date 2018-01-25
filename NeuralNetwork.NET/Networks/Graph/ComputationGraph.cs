using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Delegates;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Exceptions;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Graph.Nodes;
using NeuralNetworkNET.Networks.Graph.Nodes.Abstract;
using NeuralNetworkNET.Networks.Layers.Abstract;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A graph of <see cref="INetworkLayer"/> instances, with O(1) pre-order access time for nodes
    /// </summary>
    [JsonConverter(typeof(ComputationGraphJsonConverter))]
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
                NodeBase next;
                switch (node.NodeType)
                {
                    case ComputationGraphNodeType.Input:
                        if (node.Parents.Count > 0) throw new ComputationGraphBuildException("An input node can't haave any parent nodes");
                        if (node.Children.Count == 0) throw new ComputationGraphBuildException("An input node can't have 0 child nodes");
                        next = new InputNode();
                        map[node] = (next, input, id);
                        break;
                    case ComputationGraphNodeType.Processing:
                        if (node.Parents.Count != 1) throw new ComputationGraphBuildException("A processing node must have a single parent node");
                        INetworkLayer layer = node.GetParameter<LayerFactory>()(map[node.Parents[0]].Info);
                        ProcessingNode processing = new ProcessingNode(layer, map[node.Parents[0]].Node);
                        if (layer is OutputLayerBase)
                        {
                            if (node.Children.Count > 0) throw new ComputationGraphBuildException("An output node can't have any child nodes");
                            if (id != default)
                            {
                                if (map.Values.Any(entry => entry.Node is ProcessingNode p && p.Layer is OutputLayerBase && entry.Id == id))
                                    throw new ComputationGraphBuildException("Each training branch can have a single output node");
                                trainingOutputs.Add(processing);
                            }
                            else if (output == null)  output = processing;
                            else throw new ComputationGraphBuildException("The graph can only have a single inference output node");
                        }
                        else if (node.Children.Count == 0) throw new ComputationGraphBuildException("A processing node can't have 0 child nodes");
                        next = processing;
                        map[node] = (next, layer.OutputInfo, id);
                        break;
                    case ComputationGraphNodeType.TrainingBranch:
                        if (id != default) throw new ComputationGraphBuildException("A training branch can't contain secondary training branches");
                        if (node.Children.Count == 0) throw new ComputationGraphBuildException("A training branch node can't have 0 child nodes");
                        if (node.Parents.Count != 1) throw new ComputationGraphBuildException("A training branch node must have a single parent node");
                        if (node.Parents[0].NodeType == ComputationGraphNodeType.Input) throw new ComputationGraphBuildException("A training branch can't start right from an input node");
                        next = new TrainingNode(map[node.Parents[0]].Node);
                        map[node] = (next, map[node.Parents[0]].Info, id);
                        id = Guid.NewGuid();
                        break;
                    case ComputationGraphNodeType.DepthConcatenation:
                    case ComputationGraphNodeType.Sum:
                        if (node.Parents.Count < 2) throw new ComputationGraphBuildException("A merge node must have at least 2 parent nodes");
                        if (node.Children.Count == 0) throw new ComputationGraphBuildException("A merge node can't have 0 child nodes");

                        // Gather the parent nodes
                        var parents = new (NodeBase Node, TensorInfo Info, Guid Id)[node.Parents.Count];
                        foreach ((NodeBuilder parent, int i) in node.Parents.Select((p, i) => (p, i)))
                            if (!map.TryGetValue(parent, out parents[i]))
                                return;
                        if (parents.Skip(1).Any(p => p.Id != parents[0].Id)) throw new ComputationGraphBuildException("Can't merge a training branch back into the main graph");

                        // Calculate the output tensor size
                        TensorInfo shape;
                        if (node.NodeType == ComputationGraphNodeType.Sum)
                        {
                            shape = parents[0].Info;
                            if (parents.Skip(1).Any(p => p.Info != shape)) throw new ComputationGraphBuildException("The inputs of a sum node must all have the same shape");
                            (ActivationFunctionType activation, ExecutionModePreference mode) = node.GetParameter<(ActivationFunctionType, ExecutionModePreference)>();
                            next = SumNode.New(activation, mode, parents.Select(t => t.Node).ToArray());
                        }
                        else
                        {
                            shape = TensorInfo.Volume(parents[0].Info.Height, parents[0].Info.Width, parents.Sum(p => p.Info.Channels));
                            next = new DepthConcatenationNode(parents.Select(t => t.Node).ToArray());
                        }
                        map[node] = (next, shape, id);
                        break;
                    default:
                        throw new ComputationGraphBuildException($"Invalid node type: {node.NodeType}");
                }
                nodes.Add(next);
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

        #region Serialization

        /// <summary>
        /// Writes the current graph to the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The target <see cref="Stream"/> to use to write the graph data</param>
        public void Serialize([NotNull] Stream stream)
        {
            /* =================
             * Data structure
             * =================
             * The input stream will contain the data needed to rebuild an identical computation
             * graph. It is not possible to directly rebuild each graaph node, since the
             * constructor method for each node type takes the list of parent nodes, which wouldn't
             * be available while the data deserialization is still being performed.
             * The approach used in this case is to divide the resulting data stream into two sections:
             * 1:   A unique Guid is assigned to each node, and the linear list of in-order nodes
             *      is serialized as a series of (Guid, node) serialized pairs. For processing
             *      nodes, the underlying network layer is serialized as well.
             * 2:   A list of child nodes for each node, using the previously
             *      assigned Guid to identify each serialized node.
             * When deserializing the graph, a NodeBuilder instance is created for each (Guid, node)
             * pair, then the data from the second section is used to reconstruct the right spatial
             * relations for each node. Finally, a new graph is built from the resulting nodes. */
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
            }
        }

        /// <summary>
        /// Tries to deserialize a new <see cref="ComputationGraph"/> from the input <see cref="Stream"/> and returns a <see cref="Func{TIn, TResult}"/> to rebuild it
        /// </summary>
        /// <param name="stream">The input <see cref="Stream"/> to use to read the network data</param>
        /// <param name="preference">The layers deserialization preference</param>
        [MustUseReturnValue, CanBeNull]
        public static Func<TensorInfo, ComputationGraph> Deserialize([NotNull] Stream stream, ExecutionModePreference preference)
        {
            // Prepare the node builders with the appropriate nodes
            if (!stream.TryRead(out int count)) return null;
            Dictionary<Guid, NodeBuilder> map = new Dictionary<Guid, NodeBuilder>();
            for (int i = 0; i < count; i++)
            {
                if (!stream.TryRead(out Guid id)) return null;
                if (!stream.TryRead(out ComputationGraphNodeType type)) return null;
                switch (type)
                {
                    case ComputationGraphNodeType.Processing:
                        if (!stream.TryRead(out LayerType layerType)) return null;
                        INetworkLayer layer = null;
                        if (preference == ExecutionModePreference.Cuda) layer = NetworkLoader.CuDnnLayerDeserialize(stream, layerType);
                        if (layer == null) layer = NetworkLoader.CpuLayerDeserialize(stream, layerType);
                        if (layer == null) return null;
                        map[id] = new NodeBuilder(type, new LayerFactory(_ => layer));
                        break;
                    case ComputationGraphNodeType.Sum:
                        if (!stream.TryRead(out ActivationFunctionType activation)) return null;
                        map[id] = new NodeBuilder(type, (activation, preference));
                        break;
                    default:
                        map[id] = new NodeBuilder(type, null);
                        break;
                }
            }

            // Assign the neighbours
            while (stream.TryRead(out Guid id))
            {
                if (!stream.TryRead(out int children)) return null;
                for (int i = 0; i < children; i++)
                {
                    if (!stream.TryRead(out Guid child)) return null;
                    map[id].Children.Add(map[child]);
                    map[child].Parents.Add(map[id]);
                }
            }
            return t => New(t, map.Values.First(node => node.NodeType == ComputationGraphNodeType.Input));
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
                    case DepthConcatenationNode merge:
                        if (!GetIndexes(merge.Parents, Nodes).SequenceEqual(GetIndexes(n2.To<IComputationGraphNode, DepthConcatenationNode>().Parents, other.Nodes))) return false;
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
        /// Creates a <see cref="Func{TIn, TResult}"/> to clone the current instance, given the input <see cref="TensorInfo"/>
        /// </summary>
        [Pure, NotNull]
        public Func<TensorInfo, ComputationGraph> GetCloneFactory()
        {
            // Build the node mapping
            Dictionary<IComputationGraphNode, NodeBuilder> map = Nodes.ToDictionary(n => n, n =>
            {
                switch (n)
                {
                    case ProcessingNode processing: return new NodeBuilder(n.Type, new LayerFactory(_ => processing.Layer.Clone()));
                    case SumNode sum: return new NodeBuilder(ComputationGraphNodeType.Sum, (sum.ActivationFunctionType, sum.ExecutionMode));
                    case DepthConcatenationNode _:
                    case TrainingNode _:
                    case InputNode _: return new NodeBuilder(n.Type, null);
                    default: throw new InvalidOperationException("Invalid graph node type");
                }
            });

            // Write the neighbours of each node
            foreach (IComputationGraphNode node in Nodes)
            {
                foreach (IComputationGraphNode child in node.Children)
                {
                    map[node].Children.Add(map[child]);
                    map[child].Parents.Add(map[node]);
                }
            }
            return t => New(t, map.Values.First(node => node.NodeType == ComputationGraphNodeType.Input));
        }

        #endregion
    }
}
