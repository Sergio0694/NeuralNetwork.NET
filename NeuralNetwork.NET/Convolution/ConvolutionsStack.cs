using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;

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
        public ConvolutionsStack Expand([NotNull] ConvolutionFunction func, params double[][,] kernels)
        {
            return this.Select(layer => kernels.Select(k => func(layer, k)).ToArray()).ToArray();
        }

        #region Implicit operators

        /// <summary>
        /// Implicitly converts an array of 2D layers to a volume (used to make the class easier to use externally)
        /// </summary>
        /// <param name="data">The source data</param>
        [PublicAPI]
        [NotNull]
        public static implicit operator ConvolutionsStack([NotNull, ItemNotNull] double[][,] data) => new ConvolutionsStack(data);

        /// <summary>
        /// Implicitly converts an series of arrays of 2D layers to a volume (used to make the class easier to use externally)
        /// </summary>
        /// <param name="data">The source data</param>
        [PublicAPI]
        [NotNull]
        public static implicit operator ConvolutionsStack([NotNull, ItemNotNull] double[][][,] data)
        {
            return new ConvolutionsStack(data.SelectMany(volume => volume).ToArray());
        }

        #endregion

        #region IEnumerable

        // Forwards the stack iterator
        public IEnumerator<double[,]> GetEnumerator() => Stack.GetEnumerator();

        // Default enumerator
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        #endregion
    }
}
