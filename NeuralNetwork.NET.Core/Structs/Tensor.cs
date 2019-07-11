using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using JetBrains.Annotations;
using NeuralNetworkDotNet.Core.Enums;
using NeuralNetworkDotNet.Core.Helpers;

namespace NeuralNetworkDotNet.Core.Structs
{
    /// <summary>
    /// A readonly struct that holds the info on an unmanaged memory area that has been allocated
    /// </summary>
    [DebuggerTypeProxy(typeof(_TensorProxy))]
    [DebuggerDisplay("N: {N}, CHW: {CHW}, Size: {Size}")]
    public readonly struct Tensor
    {
        /// <summary>
        /// The number of rows in the current <see cref="Tensor"/>
        /// </summary>
        public readonly int N;

        /// <summary>
        /// The size of the CHW channels in the current <see cref="Tensor"/>
        /// </summary>
        public readonly int CHW;

        /// <summary>
        /// The total size (the number of <see cref="float"/> values) in the current <see cref="Tensor"/>
        /// </summary>
        public int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => N * CHW;
        }

        /// <summary>
        /// The <see cref="float"/> buffer with the underlying data for the current instance
        /// </summary>
        [NotNull]
        private readonly float[] Data;

        /// <summary>
        /// Gets a <see cref="Span{T}"/> instance for the current <see cref="Tensor"/> data
        /// </summary>
        public Span<float> Span
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Data.AsSpan(0, Size);
        }

        // Private constructor
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Tensor([NotNull] float[] data, int n, int chw)
        {
            Data = data;
            N = n;
            CHW = chw;
        }

        /// <summary>
        /// Creates a new <see cref="Tensor"/> instance with the specified shape
        /// </summary>
        /// <param name="n">The height of the <see cref="Tensor"/></param>
        /// <param name="chw">The width of the <see cref="Tensor"/></param>
        /// <param name="mode">The desired allocation mode to use when creating the new <see cref="Tensor"/> instance</param>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Tensor New(int n, int chw, AllocationMode mode = AllocationMode.Default)
        {
            Guard.IsTrue(n > 0, nameof(n), "N must be a positive number");
            Guard.IsTrue(chw > 0, nameof(chw), "CHW must be a positive number");

            var data = ArrayPool<float>.Shared.Rent(n * chw);
            var tensor = new Tensor(data, n, chw);
            if (mode == AllocationMode.Clean) tensor.Span.Clear();

            return tensor;
        }

        /// <summary>
        /// Creates a new instance with the same shape as the input <see cref="Tensor"/>
        /// </summary>
        /// <param name="tensor">The <see cref="Tensor"/> to use to copy the shape</param>
        /// <param name="mode">The desired allocation mode to use when creating the new <see cref="Tensor"/> instance</param>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Tensor Like(in Tensor tensor, AllocationMode mode = AllocationMode.Default) => New(tensor.N, tensor.CHW, mode);

        /// <summary>
        /// Creates a new instance by copying the contents of the input vector and reshaping it to the desired size
        /// </summary>
        /// <param name="v">The input vector to copy</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Tensor From([NotNull] float[] v)
        {
            Guard.IsTrue(v.Length >= 0, nameof(v), "The input vector can't be empty");

            return new Tensor(v, 1, v.Length);
        }

        /// <summary>
        /// Creates a new instance by copying the contents of the input vector and reshaping it to the desired size
        /// </summary>
        /// <param name="v">The input vector to copy</param>
        /// <param name="n">The height of the final <see cref="Tensor"/></param>
        /// <param name="chw">The width of the final <see cref="Tensor"/></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Tensor From([NotNull] float[] v, int n, int chw)
        {
            Guard.IsTrue(v.Length >= 0, nameof(v), "The input vector can't be empty");
            Guard.IsTrue(v.Length == n * chw, "The input shape doesn't match the size of the input vector");

            return new Tensor(v, n, chw);
        }

        /// <summary>
        /// Creates a new instance by copying the contents of the input matrix and reshaping it to the desired size
        /// </summary>
        /// <param name="m">The input matrix to copy</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe Tensor From([NotNull] float[,] m)
        {
            Guard.IsTrue(m.Length >= 0, nameof(m), "The input matrix can't be empty");

            var tensor = New(m.GetLength(0), m.GetLength(1));
            fixed (float* p = m)
            {
                new Span<float>(p, m.Length).CopyTo(tensor.Span);
            }

            return tensor;
        }

        /// <summary>
        /// Creates a new instance by wrapping the current memory area
        /// </summary>
        /// <param name="n">The height of the final <see cref="Tensor"/></param>
        /// <param name="chw">The width of the final <see cref="Tensor"/></param>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Tensor Reshape(int n, int chw)
        {
            Guard.IsTrue(n * chw == Size, "The input reshaped size is invalid");

            return new Tensor(Data, n, chw);
        }

        /// <summary>
        /// Overwrites the contents of the current instance with the input <see cref="Tensor"/>
        /// </summary>
        /// <param name="tensor">The input <see cref="Tensor"/> to copy</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Overwrite(in Tensor tensor)
        {
            Guard.IsTrue(N == tensor.N, nameof(N), "The N parameter in the input tensor doesn't match the current one");
            Guard.IsTrue(CHW == tensor.CHW, nameof(CHW), "The CHW parameter in the input tensor doesn't match the current one");

            tensor.Span.CopyTo(Span);
        }

        /// <summary>
        /// Duplicates the current instance to an output <see cref="Tensor"/>
        /// </summary>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Tensor Duplicate()
        {
            var tensor = New(N, CHW);
            Span.CopyTo(tensor.Span);

            return tensor;
        }

        #region Debug

        /// <summary>
        /// A proxy type to debug instances of the <see cref="Tensor"/> <see langword="struct"/>
        /// </summary>
        private readonly struct _TensorProxy
        {
            /// <summary>
            /// Gets a preview of the underlying memory area wrapped by this instance
            /// </summary>
            [NotNull]
            [SuppressMessage("ReSharper", "MemberCanBePrivate.Local")]
            [SuppressMessage("ReSharper", "UnusedAutoPropertyAccessor.Local")]
            public IEnumerable<float[]> RowsPreview { get; }

            /// <summary>
            /// The maximum number of rows to display in the debugger
            /// </summary>
            private const int MaxRows = 10;

            /// <summary>
            /// The maximum number of total items to display in the debugger
            /// </summary>
            private const int MaxItems = 30000;

            [SuppressMessage("ReSharper", "UnusedMember.Local")]
            public _TensorProxy(Tensor obj)
            {
                // Iterator to delay the creation of the debugger display rows until requested by the user
                IEnumerable<float[]> ExtractRows()
                {
                    int
                        cappedRows = MaxItems / obj.CHW,
                        rows = Math.Min(MaxRows, cappedRows);
                    for (int i = 0; i < rows; i++)
                        yield return obj.Span.Slice(i * obj.CHW, obj.CHW).ToArray();
                }

                RowsPreview = ExtractRows();
            }
        }

        #endregion
    }
}
