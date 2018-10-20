using System;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Exceptions
{
    /// <summary>
    /// A class that represents an error during the setup of a neural network
    /// </summary>
    public sealed class NetworkBuildException : ArgumentException
    {
        internal NetworkBuildException([NotNull] string message) : base(message) { }

        internal NetworkBuildException([NotNull] string message, [NotNull] string param) : base(message, param) { }
    }
}
