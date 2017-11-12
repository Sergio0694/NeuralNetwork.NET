using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Exceptions
{
    /// <summary>
    /// An <see cref="Exception"/> that is raised whenever the execution of a parallel code block fails
    /// </summary>
    public sealed class ParallelLoopExecutionErrorException : InvalidOperationException
    {
        internal ParallelLoopExecutionErrorException() : base("Error while performing the parallel loop") { }
    }

    /// <summary>
    /// A simple class that contains some extension methods for the <see cref="Parallel"/> class
    /// </summary>
    internal static class ParallelLoopExtensions
    {
        /// <summary>
        /// Raises a <see cref="ParallelLoopExecutionErrorException"/> if the loop wasn't completed successfully
        /// </summary>
        /// <param name="result">The <see cref="ParallelLoopResult"/> to test</param>
        [AssertionMethod]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AssertCompleted(this ParallelLoopResult result)
        {
            if (!result.IsCompleted) throw new ParallelLoopExecutionErrorException();
        }
    }
}
