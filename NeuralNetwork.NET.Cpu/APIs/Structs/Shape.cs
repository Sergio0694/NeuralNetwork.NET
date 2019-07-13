using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using NeuralNetworkDotNet.Helpers;

namespace NeuralNetworkDotNet.APIs.Structs
{
    /// <summary>
    /// A <see langword="struct"/> that contains info on the size of a given tensor
    /// </summary>
    [DebuggerDisplay("[{N}, {C}, {H}, {W}], size: {NCHW}")]
    public readonly struct Shape : IEquatable<Shape>
    {
        /// <summary>
        /// The N dimension (samples) of the current <see cref="Shape"/> instance
        /// </summary>
        public readonly int N;

        /// <summary>
        /// The C dimension (channels) of the current <see cref="Shape"/> instance
        /// </summary>
        public readonly int C;

        /// <summary>
        /// The H dimension (height) of the current <see cref="Shape"/> instance
        /// </summary>
        public readonly int H;

        /// <summary>
        /// The W dimension (width) of the current <see cref="Shape"/> instance
        /// </summary>
        public readonly int W;

        /// <summary>
        /// Gets the total size (the number of <see cref="float"/> values) in the current <see cref="Shape"/> instance
        /// </summary>
        public int NCHW
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                if (N == -1) return CHW;
                return N * C * H * W;
            }
        }

        /// <summary>
        /// Gets the CHW size (the number of <see cref="float"/> values) in the current <see cref="Shape"/> instance
        /// </summary>
        public int CHW
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => C * H * W;
        }

        /// <summary>
        /// Gets the HW size (the number of <see cref="float"/> values) in the current <see cref="Shape"/> instance
        /// </summary>
        public int HW
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => H * W;
        }

        /// <summary>
        /// Creates a new <see cref="Shape"/> instance with the provided parameters
        /// </summary>
        /// <param name="n">The N dimension (samples) of the <see cref="Shape"/></param>
        /// <param name="c">The C dimension (channels) of the <see cref="Shape"/></param>
        /// <param name="h">The H dimension (height) of the <see cref="Shape"/></param>
        /// <param name="w">The W dimension (width) of the <see cref="Shape"/></param>
        private Shape(int n, int c, int h, int w)
        {
            Guard.IsTrue(n == -1 || n > 0, nameof(n), "N must be either -1 or a positive number");
            Guard.IsTrue(c > 0, nameof(c), "C must be a positive number");
            Guard.IsTrue(h > 0, nameof(h), "H must be a positive number");
            Guard.IsTrue(w > 0, nameof(w), "W must be a positive number");

            N = n;
            C = c;
            H = h;
            W = w;
        }

        /// <summary>
        /// Converts a tuple to a <see cref="Shape"/> instance
        /// </summary>
        /// <param name="shape">The tuple with the values to convert</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator Shape((int N, int C, int H, int W) shape) => new Shape(shape.N, shape.C, shape.H, shape.W);

        /// <summary>
        /// Converts a tuple to a <see cref="Shape"/> instance
        /// </summary>
        /// <param name="shape">The tuple with the values to convert</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator Shape((int C, int H, int W) shape) => new Shape(-1, shape.C, shape.H, shape.W);

        #region IEquatable<Shape>

        /// <inheritdoc/>
        public bool Equals(Shape other) => this == other;

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is Shape other && Equals(other);

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            Span<int> values = stackalloc int[] { N, C, H, W };
            return values.GetContentHashCode();
        }

        /// <summary>
        /// Checks whether or not two <see cref="Shape"/> instances have the same parameters
        /// </summary>
        /// <param name="a">The first instance</param>
        /// <param name="b">The second instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator ==(in Shape a, in Shape b) => a.N == b.N && a.C == b.C && a.H == b.H && a.W == b.W;

        /// <summary>
        /// Checks whether or not two <see cref="Shape"/> instances have different parameters
        /// </summary>
        /// <param name="a">The first instance</param>
        /// <param name="b">The second instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator !=(in Shape a, in Shape b) => !(a == b);

        #endregion
    }
}
