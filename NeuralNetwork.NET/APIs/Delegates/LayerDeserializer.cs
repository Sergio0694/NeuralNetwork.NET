using System.IO;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.APIs.Delegates
{
    /// <summary>
    /// A <see cref="delegate"/> that tries to deserialize a network layer from the input <see cref="Stream"/>, assuming the layer is of the given <see cref="LayerType"/>
    /// </summary>
    /// <param name="stream">The source <see cref="Stream"/> to load data from. If the layer type is not supported, the <see cref="Stream"/> should not be read at all</param>
    /// <param name="type">The type of network layer to deserialize from the <see cref="Stream"/></param>
    [CanBeNull]
    public delegate INetworkLayer LayerDeserializer([NotNull] Stream stream, LayerType type);
}
