using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.Networks.Graph.Nodes;
using NeuralNetworkNET.Networks.Graph.Nodes.Abstract;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A simple class that handles the Json srializaation of the metadata of aa given <see cref="ComputationGraph"/> instance
    /// </summary>
    internal sealed class ComputationGraphJsonConverter : JsonConverter
    {
        /// <inheritdoc/>
        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            // Extract the nodes info
            if (!(value is ComputationGraph graph)) throw new InvalidOperationException("Invalid value to serialize");
            Dictionary<IComputationGraphNode, int> map = graph.Nodes.Select((n, i) => (Node: n, Index: i)).ToDictionary(p => p.Node, p => p.Index);
            IList<JObject> nodes = graph.Nodes.Select(node =>
            {
                // Base properties
                JObject jNode = new JObject
                {
                    ["Id"] = map[node],
                    ["Type"] = node.Type.ToString(),
                    ["Children"] = new JArray(node.Children.Select(child => map[child]).ToList())
                };

                // Node-specific properties
                switch (node)
                {
                    case ProcessingNode processing:
                        jNode.Add("Parent", map[processing.Parent]);
                        jNode.Add("Layer", JToken.FromObject(processing.Layer));
                        break;
                    case MergeNodeBase merge:
                        jNode.Add("Parents", new JArray(merge.Parents.Select(child => map[child]).ToList()));
                        break;
                    case TrainingNode split:
                        jNode.Add("Parent", map[split.Parent]);
                        break;
                    case InputNode _: break;
                    default: throw new ArgumentException("Invalid node type to serialize");
                }
                return jNode;
            }).ToList();

            // Serialize the graph
            JObject jObj = new JObject
            {
                ["Size"] = graph.Nodes.Count,
                ["Auxiliary classifiers"] = graph.TrainingOutputNodes.Count,
                ["Nodes"] = new JArray(nodes)
            };
            jObj.WriteTo(writer);
        }

        /// <inheritdoc/>
        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
            => throw new NotSupportedException();

        /// <inheritdoc/>
        public override bool CanConvert(Type objectType) => objectType == typeof(ComputationGraph);
    }
}
