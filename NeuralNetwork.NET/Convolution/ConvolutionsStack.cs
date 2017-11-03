using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Convolution.Delegates;

namespace NeuralNetworkNET.Convolution
{
    /// <summary>
    /// A class that represents a volume of data resulting from a convolution pipeline executed on a 2D input
    /// </summary>
    public sealed class ConvolutionsStack : IReadOnlyList<double[,]>
    {
        #region Parameters

        // The actual 3D stack
        private readonly IReadOnlyList<double[,]> Stack;

        /// <summary>
        /// Gets the depth of the convolutions volume
        /// </summary>
        [PublicAPI]
        public int Count { get; }

        /// <summary>
        /// Gets the height of the convolutions volume
        /// </summary>
        [PublicAPI]
        public int Height { get; }

        /// <summary>
        /// Gets the width of the convolutions volume
        /// </summary>
        [PublicAPI]
        public int Width { get; }

        #endregion

        // Internal constructor
        internal ConvolutionsStack([NotNull, ItemNotNull] IReadOnlyList<double[,]> stack)
        {
            if (stack.Count == 0 || stack[0].Length == 0) throw new ArgumentOutOfRangeException("The volume can't be empty");
            Stack = stack;
            Count = stack.Count;
            Height = stack[0].GetLength(0);
            Width = stack[0].GetLength(1);
        }

        /// <summary>
        /// Converts a single 2D matrix to a volume with no depth ready for further processing
        /// </summary>
        /// <param name="data">The input layer</param>
        [PublicAPI]
        [NotNull]
        public static ConvolutionsStack From2DLayer([NotNull] double[,] data) => new ConvolutionsStack(new[] { data });

        /// <summary>
        /// Gets the value in the target position inside the data volume
        /// </summary>
        /// <param name="z">The target depth, that is, the index of the target 2D layer</param>
        /// <param name="x">The horizontal offset in the 2D layer</param>
        /// <param name="y">The vertical offset in the target 2D layer</param>
        [PublicAPI]
        public double this[int z, int x, int y] => Stack[z][x, y];

        /// <summary>
        /// Gets the 2D layer at the target depth in the data volume
        /// </summary>
        /// <param name="z">The depth of the target 2D layer to retrieve</param>
        [PublicAPI]
        [NotNull]
        public double[,] this[int z] => Stack[z];

        /// <summary>
        /// Expands the curret data volume with the input convolution function and a series of convolution kernels
        /// </summary>
        /// <param name="func">The convolution function to use</param>
        /// <param name="kernels">The convolution kernels</param>
        /// <remarks>The resulting volume will have a depth equals to the current one multiplied by the number of kernels used</remarks>
        [PublicAPI]
        [Pure, NotNull]
        internal ConvolutionsStack Expand([NotNull] ConvolutionFunction func, params double[][,] kernels)
        {
            double[][][,] expansion = this.Select(layer => kernels.Select(k => func(layer, k)).ToArray()).ToArray();
            double[][,] stack = expansion.SelectMany(volume => volume).ToArray();
            return new ConvolutionsStack(stack);
        }

        /// <summary>
        /// Processes each depth layer in the current stack with the given function and returns a new data volume
        /// </summary>
        /// <param name="processor">The data processor to use for each data layer in the stack</param>
        [PublicAPI]
        [Pure, NotNull]
        internal ConvolutionsStack Process([NotNull] LayerProcessor processor)
        {
            double[][,] data = this.Select(layer => processor(layer)).ToArray();
            return new ConvolutionsStack(data);
        }

        /// <summary>
        /// Flattens a vector of data volumes into a single 2D matrix
        /// </summary>
        /// <param name="data">The data to convert</param>
        [PublicAPI]
        [Pure]
        public static double[,] ConvertToMatrix([NotNull, ItemNotNull] params ConvolutionsStack[] data)
        {
            // Checks
            if (data.Length == 0) throw new ArgumentOutOfRangeException("The data array can't be empty");

            // Prepare the base network and the input data
            int
                depth = data[0].Count, // Depth of each convolution volume
                ch = data[0].Height, // Height of each convolution layer
                cw = data[0].Width, // Width of each convolution layer
                lsize = ch * cw,
                volume = depth * lsize;

            // Additional checks
            if (data.Any(stack => stack.Count != depth || stack.Height != ch || stack.Width != cw))
                throw new ArgumentException("The input data isn't coherent");

            // Setup the matrix with all the batched inputs
            double[,] x = new double[data.Length, volume];

            // Populate the matrix, iterate over all the volumes
            bool result = Parallel.For(0, data.Length, i =>
            {
                unsafe
                {
                    // Fix the pointers
                    fixed (double* px = x)
                    {
                        ConvolutionsStack stack = data[i];
                        for (int j = 0; j < depth; j++) // Iterate over all the depth layer in each volume
                            for (int z = 0; z < ch; z++) // Height of each layer
                                for (int w = 0; w < cw; w++) // Width of each layer
                                    px[i * volume + j * lsize + z * ch + w] = stack[j, z, w];
                    }
                }
            }).IsCompleted;
            if (!result) throw new Exception("Error while running the parallel loop");
            return x;
        }

        #region IEnumerable

        // Forwards the stack iterator
        public IEnumerator<double[,]> GetEnumerator() => Stack.GetEnumerator();

        // Default enumerator
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        #endregion
    }
}
