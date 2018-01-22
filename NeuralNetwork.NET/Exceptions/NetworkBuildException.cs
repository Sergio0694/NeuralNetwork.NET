using System;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Exceptions
{
    /// <summary>
    /// A class that represents an error during the setup of a neural network
    /// </summary>
    public sealed class NetworkBuildException : ArgumentException
    {
        internal NetworkBuildException([NotNull] String message) : base(message) { }

        internal NetworkBuildException([NotNull] String message, [NotNull] String param) : base(message, param) { }
    }
}
