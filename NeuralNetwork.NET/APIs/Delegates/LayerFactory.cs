using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;

namespace NeuralNetworkNET.APIs.Delegates
{
    /// <summary>
    /// A <see langword="delegate"/> that represents a factory that produces instances of a specific layer type, with user-defined parameters.
    /// This wrapper acts as an intemediary to streamline the user-side C# sintax when building up a new network structure, as all the input
    /// details for each layer will be automatically computed during the network setup.
    /// </summary>
    /// <param name="info">The <see cref="TensorInfo"/> for the inputs of the upcoming network layer</param>
    /// <remarks>It is also possible to invoke a <see cref="LayerFactory"/> instance just like any other <see langword="delegate"/> to immediately get an <see cref="INetworkLayer"/> value</remarks>
    [NotNull]
    public delegate INetworkLayer LayerFactory(TensorInfo info);
}
