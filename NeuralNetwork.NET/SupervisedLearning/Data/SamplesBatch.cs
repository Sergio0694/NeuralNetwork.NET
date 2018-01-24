using System;
using System.Collections.Generic;
using System.Diagnostics;
using JetBrains.Annotations;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.SupervisedLearning.Data
{
    /// <summary>
    /// A simple struct that keeps a reference of a training set and its expected results
    /// </summary>
    [DebuggerDisplay("Samples: {X.GetLength(0)}, inputs: {X.GetLength(1)}, outputs: {Y.GetLength(1)}")]
    internal readonly struct SamplesBatch
    {
        /// <summary>
        /// Gets the current dataset
        /// </summary>
        [NotNull]
        public readonly float[,] X;

        /// <summary>
        /// Gets the expected results for the current dataset
        /// </summary>
        [NotNull]
        public readonly float[,] Y;

        /// <summary>
        /// Creates a new dataset wrapper batch with the given data
        /// </summary>
        /// <param name="x">The batch data</param>
        /// <param name="y">The batch expected results</param>
        public SamplesBatch([NotNull] float[,] x, [NotNull] float[,] y)
        {
            if (x.GetLength(0) != y.GetLength(0)) throw new ArgumentException("The number of samples in the data and results must be the same");
            X = x;
            Y = y;
        }

        /// <summary>
        /// Creates a new instance from the input partition
        /// </summary>
        /// <param name="x">A <see cref="Span{T}"/> representing the input samples partition</param>
        /// <param name="y">A <see cref="Span{T}"/> representing the output samples partition</param>
        /// <param name="inputs">The number of input features</param>
        /// <param name="outputs">The number of output features</param>
        public static SamplesBatch From(Span<float> x, Span<float> y, int inputs, int outputs)
        {
            int samples = x.Length / inputs;
            if (y.Length / outputs != samples) throw new ArgumentException("The size of the input and output datasets isn't valid");
            return new SamplesBatch(x.AsMatrix(samples, inputs), y.AsMatrix(samples, outputs));
        }

        /// <summary>
        /// Creates a new instance from the input partition
        /// </summary>
        /// <param name="batch">The source batch</param>
        public static SamplesBatch From([NotNull] IReadOnlyList<(float[] X, float[] Y)> batch)
        {
            int
                wx = batch[0].X.Length,
                wy = batch[0].Y.Length;
            float[,]
                xBatch = new float[batch.Count, wx],
                yBatch = new float[batch.Count, wy];
            for (int i = 0; i < batch.Count; i++)
            {
                Buffer.BlockCopy(batch[i].X, 0, xBatch, sizeof(float) * i * wx, sizeof(float) * wx);
                Buffer.BlockCopy(batch[i].Y, 0, yBatch, sizeof(float) * i * wy, sizeof(float) * wy);
            }
            return new SamplesBatch(xBatch, yBatch);
        }

        /// <summary>
        /// Creates a new instance from the input partition
        /// </summary>
        /// <param name="batch">The source batch</param>
        /// <param name="inputs">The number of input features</param>
        /// <param name="outputs">The number of output features</param>
        public static unsafe SamplesBatch From([NotNull] IReadOnlyList<(Pin<float>, Pin<float>)> batch, int inputs, int outputs)
        {
            float[,]
                xBatch = new float[batch.Count, inputs],
                yBatch = new float[batch.Count, outputs];
            uint
                wx = (uint)(sizeof(float) * inputs),
                wy = (uint)(sizeof(float) * outputs);
            fixed (float* px = xBatch, py = yBatch)
            {
                for (int i = 0; i < batch.Count; i++)
                {
                    var (x, y) = batch[i];
                    Buffer.MemoryCopy(x.Ptr, px + i * inputs, wx, wx);
                    Buffer.MemoryCopy(y.Ptr, py + i * outputs, wy, wy);
                }
            }
            return new SamplesBatch(xBatch, yBatch);
        }
    }
}