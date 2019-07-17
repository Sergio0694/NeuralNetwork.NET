using System;
using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Interfaces;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.Helpers;

namespace NeuralNetworkDotNet.APIs.Models
{
    /// <summary>
    /// A readonly struct that holds the info on an unmanaged memory area that has been allocated
    /// </summary>
    [DebuggerDisplay("Shape: {" + nameof(Shape) + "}")]
    public sealed class Tensor : IDisposable, IEquatable<Tensor>, IClonable<Tensor>
    {
        /// <summary>
        /// Gets the shape of the current <see cref="Tensor"/> instance
        /// </summary>
        public readonly Shape Shape;

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
            get => Data.AsSpan(0, Shape.NCHW);
        }

        /// <summary>
        /// Gets a <see cref="Span{T}"/> instance for a specific sample in the current <see cref="Tensor"/>
        /// </summary>
        /// <param name="n">The index of the item to retrieve</param>
        public Span<float> this[int n]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                var offset = n * Shape.CHW;
                return Data.AsSpan(offset, Shape.CHW);
            }
        }

        /// <summary>
        /// Gets a <see cref="Span{T}"/> instance for a specific 2D slice in the current <see cref="Tensor"/>
        /// </summary>
        /// <param name="n">The index of the item to retrieve</param>
        /// <param name="c">The index of the channel to retrieve</param>
        public Span<float> this[int n, int c]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                var offset = n * Shape.CHW + c * Shape.HW;
                return Data.AsSpan(offset, Shape.HW);
            }
        }

        /// <summary>
        /// Gets a <see cref="Span{T}"/> instance for a specific row slice in the current <see cref="Tensor"/>
        /// </summary>
        /// <param name="n">The index of the item to retrieve</param>
        /// <param name="c">The index of the channel to retrieve</param>
        /// <param name="h">The index of the row to retrieve</param>
        public Span<float> this[int n, int c, int h]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                var offset = n * Shape.CHW + c * Shape.HW + h * Shape.W;
                return Data.AsSpan(offset, Shape.W);
            }
        }

        /// <summary>
        /// Gets a reference to a specifc value in the current <see cref="Tensor"/>
        /// </summary>
        /// <param name="n">The index of the item to retrieve</param>
        /// <param name="c">The index of the channel to retrieve</param>
        /// <param name="h">The index of the row to retrieve</param>
        /// <param name="w">The index of the column to retrieve</param>
        public ref float this[int n, int c, int h, int w]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                var offset = n * Shape.CHW + c * Shape.HW + h * Shape.W + w;
                return ref Data[offset];
            }
        }

        // Private constructor
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Tensor(Shape shape, [NotNull] float[] data)
        {
            Shape = shape;
            Data = data;
        }

        /// <summary>
        /// Creates a new <see cref="Tensor"/> instance with the specified shape
        /// </summary>
        /// <param name="n">The N dimension (samples) of the <see cref="Tensor"/></param>
        /// <param name="l">The number of values for each sample of the <see cref="Tensor"/></param>
        /// <param name="mode">The desired allocation mode to use when creating the new <see cref="Tensor"/> instance</param>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Tensor New(int n, int l, AllocationMode mode = AllocationMode.Default) => New((n, 1, 1, l), mode);

        /// <summary>
        /// Creates a new <see cref="Tensor"/> instance with the specified shape
        /// </summary>
        /// <param name="n">The N dimension (samples) of the <see cref="Tensor"/></param>
        /// <param name="c">The C dimension (channels) of the <see cref="Tensor"/></param>
        /// <param name="h">The H dimension (height) of the <see cref="Tensor"/></param>
        /// <param name="w">The W dimension (width) of the <see cref="Tensor"/></param>
        /// <param name="mode">The desired allocation mode to use when creating the new <see cref="Tensor"/> instance</param>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Tensor New(int n, int c, int h, int w, AllocationMode mode = AllocationMode.Default) => New((n, c, h, w), mode);

        /// <summary>
        /// Creates a new <see cref="Tensor"/> instance with the specified shape
        /// </summary>
        /// <param name="shape">The desired shape for the new <see cref="Tensor"/></param>
        /// <param name="mode">The desired allocation mode to use when creating the new <see cref="Tensor"/> instance</param>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Tensor New(Shape shape, AllocationMode mode = AllocationMode.Default)
        {
            var data = ArrayPool<float>.Shared.Rent(shape.NCHW);
            var tensor = new Tensor(shape, data);
            if (mode == AllocationMode.Clean) tensor.Span.Clear();

            return tensor;
        }

        /// <summary>
        /// Creates a new instance with the same shape as the input <see cref="Tensor"/>
        /// </summary>
        /// <param name="source">The <see cref="Tensor"/> to use to copy the shape</param>
        /// <param name="mode">The desired allocation mode to use when creating the new <see cref="Tensor"/> instance</param>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Tensor Like([NotNull] Tensor source, AllocationMode mode = AllocationMode.Default) => New(source.Shape, mode);

        /// <summary>
        /// Creates a new instance by copying the contents of the input vector and reshaping it to the desired size
        /// </summary>
        /// <param name="v">The input vector to copy</param>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Tensor From([NotNull] float[] v)
        {
            Guard.IsTrue(v.Length >= 0, nameof(v), "The input vector can't be empty");

            var tensor = New(1, v.Length);
            v.AsSpan().CopyTo(tensor.Span);
            return tensor;
        }

        /// <summary>
        /// Creates a new instance by copying the contents of the input vector and reshaping it to the desired size
        /// </summary>
        /// <param name="v">The input vector to copy</param>
        /// <param name="n">The N dimension (samples) of the <see cref="Tensor"/></param>
        /// <param name="c">The C dimension (channels) of the <see cref="Tensor"/></param>
        /// <param name="h">The H dimension (height) of the <see cref="Tensor"/></param>
        /// <param name="w">The W dimension (width) of the <see cref="Tensor"/></param>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Tensor From([NotNull] float[] v, int n, int c, int h, int w)
        {
            Guard.IsTrue(v.Length >= 0, nameof(v), "The input vector can't be empty");

            var tensor = New(n, c, h, w);
            v.AsSpan().CopyTo(tensor.Span);
            return tensor;
        }

        /// <summary>
        /// Creates a new instance by copying the contents of the input matrix and reshaping it to the desired size
        /// </summary>
        /// <param name="m">The input matrix to copy</param>
        [Pure, NotNull]
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
        /// Creates a new instance by copying the contents of the input matrix and reshaping it to the desired size
        /// </summary>
        /// <param name="m">The input matrix to copy</param>
        /// <param name="c">The C dimension (channels) of the <see cref="Tensor"/></param>
        /// <param name="h">The H dimension (height) of the <see cref="Tensor"/></param>
        /// <param name="w">The W dimension (width) of the <see cref="Tensor"/></param>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe Tensor From([NotNull] float[,] m, int c, int h, int w)
        {
            Guard.IsTrue(m.Length >= 0, nameof(m), "The input matrix can't be empty");
            Guard.IsTrue(m.GetLength(1) == c * h * w, "The input shape doesn't match the size of the given matrix");

            var tensor = New(m.GetLength(0), c, h, w);
            fixed (float* p = m)
            {
                new Span<float>(p, m.Length).CopyTo(tensor.Span);
            }

            return tensor;
        }

        /// <summary>
        /// Reshapes the current instance to the specified shape
        /// </summary>
        /// <param name="n">The height of the final <see cref="Tensor"/></param>
        /// <param name="l">The number of values for each sample of the <see cref="Tensor"/></param>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Tensor Reshape(int n, int l)
        {
            Guard.IsTrue(n * l == Shape.NCHW, "The input reshaped size is invalid");

            return new Tensor((n, 1, 1, l), Data);
        }

        /// <summary>
        /// Reshapes the current instance to the specified shape
        /// </summary>
        /// <param name="n">The N dimension (samples) of the <see cref="Tensor"/></param>
        /// <param name="c">The C dimension (channels) of the <see cref="Tensor"/></param>
        /// <param name="h">The H dimension (height) of the <see cref="Tensor"/></param>
        /// <param name="w">The W dimension (width) of the <see cref="Tensor"/></param>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Tensor Reshape(int n, int c, int h, int w)
        {
            Guard.IsTrue(n * c * h * w == Shape.NCHW, "The input reshaped size is invalid");

            return new Tensor((n, c, h, w), Data);
        }

        /// <summary>
        /// Overwrites the contents of the current instance with the input <see cref="Tensor"/>
        /// </summary>
        /// <param name="tensor">The input <see cref="Tensor"/> to copy</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Overwrite([NotNull] Tensor tensor)
        {
            Guard.IsTrue(tensor.Shape == Shape, nameof(tensor), "The shape of the input tensor doesn't match the current shape");

            tensor.Span.CopyTo(Span);
        }

        #region Interfaces

        /// <inheritdoc/>
        public void Dispose() => ArrayPool<float>.Shared.Return(Data);

        /// <inheritdoc/>
        public bool Equals(Tensor other)
        {
            if (other == null) return false;
            if (other.Shape != Shape) return false;

            return Span.ContentEquals(other.Span);
        }

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is Tensor other && Equals(other);

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            Span<int> hashes = stackalloc int[] { Shape.GetHashCode(), Span.GetContentHashCode() };
            return hashes.GetContentHashCode();
        }

        /// <inheritdoc/>
        public Tensor Clone()
        {
            var copy = New(Shape);
            copy.Overwrite(this);

            return copy;
        }

        #endregion
    }
}
