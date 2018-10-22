using System;

namespace NeuralNetworkNET.APIs.Structs
{
    /// <summary>
    /// A stack-only struct containing two <see cref="Span{T}"/> instances pointing to a single dataset sample
    /// </summary>
    public readonly ref struct DatasetSample
    {
        /// <summary>
        /// Gets the <see cref="Span{T}"/> referencing the current sample inputs
        /// </summary>
        public ReadOnlySpan<float> X { get; }

        /// <summary>
        /// Gets the <see cref="Span{T}"/> referencing the current sample expected outputs
        /// </summary>
        public ReadOnlySpan<float> Y { get; }

        internal DatasetSample(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
        {
            X = x;
            Y = y;
        }
    }
}
