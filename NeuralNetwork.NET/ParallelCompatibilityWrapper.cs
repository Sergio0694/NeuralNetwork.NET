using System;
using JetBrains.Annotations;

namespace NeuralNetwork.NET
{
    /// <summary>
    /// A delegate that forwards the call to the Parallel.For method and returns the value of
    /// the returned ParallelLoopResult.IsCompleted property
    /// </summary>
    /// <param name="from">The starting index</param>
    /// <param name="to">The end index (not included)</param>
    /// <param name="body">The action to run in parallel</param>
    public delegate bool ParallelFor(int from, int to, [NotNull] Action<int> body);

    /// <summary>
    /// COMPATIBILITY LAYER: interface to wrap a call to the Parallel.For method in the System.Threading.Tasks namespace
    /// </summary>
    public static class ParallelCompatibilityWrapper
    {
        /// <summary>
        /// Initializes the wrapper with the input delegate that forwards the call to Parallel.For
        /// </summary>
        /// <param name="instance">The delegate instance</param>
        public static void Initialize([NotNull] ParallelFor instance) => Instance = instance;

        /// <summary>
        /// The Parallel.For wrapper instance to use in the library (until .NET Standard 2.0 is released)
        /// </summary>
        [NotNull]
        internal static ParallelFor Instance { get; private set; } = (start, end, body) => throw new NotImplementedException();
    }
}
