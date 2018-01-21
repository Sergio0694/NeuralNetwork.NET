using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Layers.Abstract;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A graph of <see cref="INetworkLayer"/> instances, with O(1) pre-order access time for nodes
    /// </summary>
    internal sealed class ComputationGraph
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
                    if (value.Id != id) throw new ArgumentException("The training branch can't cross other graph branches");
                    return;
                }

                // Process the current node
                NodeBase iNode;
                switch (node.NodeType)
                {
                    case ComputationGraphNodeType.Input:
                        if (node.Parents.Count > 0) throw new ArgumentException("An input node can't haave any parent nodes");
                        if (node.Children.Count == 0) throw new ArgumentException("An input node can't have 0 child nodes");
                        iNode = new InputNode();
                        map[node] = (iNode, input, id);
                        break;
                    case ComputationGraphNodeType.Processing:
                        if (node.Factory == null) throw new InvalidOperationException("Missing layer factory");
                        INetworkLayer layer = node.Factory(map[node.Parents[0]].Info);
                        ProcessingNode processing = new ProcessingNode(layer, map[node.Parents[0]].Node);
                        if (layer is OutputLayerBase)
                        {
                            if (output != null) throw new ArgumentException("The graph can only have a single inference output node");
                            if (node.Children.Count > 0) throw new ArgumentException("An output node can't have any child nodes");
                            if (id != default)
                            {
                                if (map.Values.Any(entry => entry.Node is ProcessingNode p && p.Layer is OutputLayerBase && entry.Id == id))
                                    throw new ArgumentException("Each training branch can have a single output node");
                                trainingOutputs.Add(processing);
                            }
                            output = processing;
                        }
                        if (node.Children.Count == 0) throw new ArgumentException("A processing node can't have 0 child nodes");
                        iNode = processing;
                        map[node] = (iNode, layer.OutputInfo, id);
                        break;
                    case ComputationGraphNodeType.TrainingBranch:
                        if (id != default) throw new ArgumentException("A training branch can't contain secondary training branches");
                        if (node.Children.Count == 0) throw new ArgumentException("A training branch node can't have 0 child nodes");
                        iNode = new TrainingNode(map[node.Parents[0]].Node);
                        map[node] = (iNode, map[node.Parents[0]].Info, id);
                        id = Guid.NewGuid();
                        break;
                    case ComputationGraphNodeType.DepthConcatenation:
                    case ComputationGraphNodeType.Sum:
                        if (node.Children.Count == 0) throw new ArgumentException("A merge node can't have 0 child nodes");

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
                        throw new ArgumentException($"Invalid node type: {node.NodeType}", nameof(root));
                }
                nodes.Add(iNode);
                foreach (NodeBuilder child in node.Children)
                    BuildMap(child, id);
            }

            // Build the graph and bind the child nodes
            BuildMap(root, default);
            if (output == null) throw new ArgumentException("The input graph doesn't have a valid output node");
            foreach (KeyValuePair<NodeBuilder, (NodeBase Node, TensorInfo Info, Guid _)> pair in map)
            {
                pair.Value.Node.Children = pair.Key.Children.Select(child => map[child].Node).ToArray();
            }
            return new ComputationGraph(map[root].Node, nodes, output, trainingOutputs);
        }

        #endregion
    }
}
