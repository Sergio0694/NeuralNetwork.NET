using System;

namespace NeuralNetworkNET.Exceptions
{
    /// <summary>
    /// A simple class that represents a runtime error during the execution of a parallel loop
    /// </summary>
    public sealed class ParallelLoopExecutionException : InvalidOperationException
    {
        internal ParallelLoopExecutionException() : base("Error while performing a parallel loop") { }
    }
}
