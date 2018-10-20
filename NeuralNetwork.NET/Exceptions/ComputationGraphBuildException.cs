using System;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Exceptions
{
    /// <summary>
    /// An exception that indicates a failure during the build process of a computation graph
    /// </summary>
    public sealed class ComputationGraphBuildException : ArgumentException
    {
        internal ComputationGraphBuildException([NotNull] string message) : base(message) { }
    }
}
