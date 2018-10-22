using JetBrains.Annotations;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.APIs.Structs
{
    /// <summary>
    /// A readonly struct that holds the info on an unmanaged memory area that has been allocated
    /// </summary>
    [PublicAPI]
    [DebuggerTypeProxy(typeof(_TensorProxy))]
    [DebuggerDisplay("Entities: {Entities}, Length: {Length}, Ptr: {Ptr}")]
    public readonly struct Tensor
    {
        #region Fields and parameters

        /// <summary>
        /// The <see cref="IntPtr"/> value to the allocated memory
        /// </summary>
        public readonly IntPtr Ptr;

        /// <summary>
        /// The number of entities (rows) in the current <see cref="Tensor"/>
        /// </summary>
        public readonly int Entities;

        /// <summary>
        /// The size of each entity in the current <see cref="Tensor"/>
        /// </summary>
        public readonly int Length;

        /// <summary>
        /// The total size (the number of <see cref="float"/> values) in the current <see cref="Tensor"/>
        /// </summary>
        public int Size
        {
            [Pure]
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Entities * Length;
        }

        /// <summary>
        /// Gets whether or not the current instance is linked to an allocated memory area
        /// </summary>
        public bool IsNull
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Ptr == IntPtr.Zero;
        }

        /// <summary>
        /// Gets a managed reference for the current <see cref="Tensor"/> data
        /// </summary>
        public unsafe ref float Ref
        {
            [PublicAPI]
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Unsafe.AsRef<float>(this);
        }

        /// <summary>
        /// Gets or sets the <see cref="Tensor"/> value for the specific index
        /// </summary>
        /// <param name="i">The target index to read or write</param>
        public unsafe float this[int i]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ((float*)this)[i];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => ((float*)this)[i] = value;
        }

        #endregion

        /// <summary>
        /// Gets a <see langword="null"/> instance
        /// </summary>
        public static readonly Tensor Null = new Tensor(IntPtr.Zero, 0, 0);

        #region Initialization

        // Private constructor
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Tensor(IntPtr ptr, int entities, int length)
        {
            Ptr = ptr;
            Entities = entities;
            Length = length;
        }

        /// <summary>
        /// Creates a new instance with the specified shape
        /// </summary>
        /// <param name="n">The height of the <see cref="Tensor"/></param>
        /// <param name="chw">The width of the <see cref="Tensor"/></param>
        /// <param name="tensor">The resulting instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void New(int n, int chw, out Tensor tensor)
        {
            IntPtr ptr = Marshal.AllocHGlobal(sizeof(float) * n * chw);
            tensor = new Tensor(ptr, n, chw);
        }

        /// <summary>
        /// Creates a new instance with the specified shape and initializes the allocated memory to 0s
        /// </summary>
        /// <param name="n">The height of the <see cref="Tensor"/></param>
        /// <param name="chw">The width of the <see cref="Tensor"/></param>
        /// <param name="tensor">The resulting instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void NewZeroed(int n, int chw, out Tensor tensor)
        {
            int bytes = sizeof(float) * n * chw;
            IntPtr ptr = Marshal.AllocHGlobal(bytes);
            tensor = new Tensor(ptr, n, chw);
            Unsafe.InitBlock(tensor, 0, (uint)bytes);
        }

        /// <summary>
        /// Creates a new instance by wrapping the input pointer
        /// </summary>
        /// <param name="p">The target memory area</param>
        /// <param name="n">The height of the final <see cref="Tensor"/></param>
        /// <param name="chw">The width of the final <see cref="Tensor"/></param>
        /// <param name="tensor">The resulting instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void Reshape(float* p, int n, int chw, out Tensor tensor)
        {
            IntPtr ptr = new IntPtr(p);
            tensor = new Tensor(ptr, n, chw);
        }

        /// <summary>
        /// Creates a new instance with the same shape as the input <see cref="Tensor"/>
        /// </summary>
        /// <param name="mask">The <see cref="Tensor"/> to use to copy the shape</param>
        /// <param name="tensor">The output <see cref="Tensor"/></param>
        public static void Like(in Tensor mask, out Tensor tensor) => New(mask.Entities, mask.Length, out tensor);

        /// <summary>
        /// Creates a new instance with the same shape as the input <see cref="Tensor"/> and all the values initializes to 0
        /// </summary>
        /// <param name="mask">The <see cref="Tensor"/> to use to copy the shape</param>
        /// <param name="tensor">The output <see cref="Tensor"/></param>
        public static void LikeZeroed(in Tensor mask, out Tensor tensor) => NewZeroed(mask.Entities, mask.Length, out tensor);

        /// <summary>
        /// Creates a new instance by copying the contents at the given memory location and reshaping it to the desired size
        /// </summary>
        /// <param name="p">The target memory area to copy</param>
        /// <param name="n">The height of the final <see cref="Tensor"/></param>
        /// <param name="chw">The width of the final <see cref="Tensor"/></param>
        /// <param name="tensor">The resulting instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void From(float* p, int n, int chw, out Tensor tensor)
        {
            New(n, chw, out tensor);
            int size = sizeof(float) * n * chw;
            Buffer.MemoryCopy(p, tensor, size, size);
        }

        /// <summary>
        /// Creates a new instance by copying the contents of the input matrix
        /// </summary>
        /// <param name="m">The input matrix to copy</param>
        /// <param name="tensor">The resulting instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void From([NotNull] float[,] m, out Tensor tensor)
        {
            fixed (float* pm = m)
                From(pm, m.GetLength(0), m.GetLength(1), out tensor);
        }

        /// <summary>
        /// Creates a new instance by copying the contents of the input vector and reshaping it to the desired size
        /// </summary>
        /// <param name="v">The input vector to copy</param>
        /// <param name="n">The height of the final <see cref="Tensor"/></param>
        /// <param name="chw">The width of the final <see cref="Tensor"/></param>
        /// <param name="tensor">The resulting instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void From([NotNull] float[] v, int n, int chw, out Tensor tensor)
        {
            if (n * chw != v.Length) throw new ArgumentOutOfRangeException(nameof(v), "The input vector doesn't have a valid size");
            fixed (float* pv = v)
                From(pv, n, chw, out tensor);
        }

        #endregion

        #region Tools

        /// <summary>
        /// Creates a new instance by wrapping the current memory area
        /// </summary>
        /// <param name="n">The height of the final <see cref="Tensor"/></param>
        /// <param name="chw">The width of the final <see cref="Tensor"/></param>
        /// <param name="tensor">The resulting instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Reshape(int n, int chw, out Tensor tensor)
        {
            if (n * chw != Size) throw new ArgumentException("Invalid input resized shape");
            tensor = new Tensor(Ptr, n, chw);
        }

        /// <summary>
        /// Checks whether or not the current instance has the same shape of the input <see cref="Tensor"/>
        /// </summary>
        /// <param name="tensor">The instance to compare</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool MatchShape(in Tensor tensor) => Entities == tensor.Entities && Length == tensor.Length;

        /// <summary>
        /// Checks whether or not the current instance has the same shape as the input arguments
        /// </summary>
        /// <param name="entities">The expected number of entities</param>
        /// <param name="length">The expected length of each entity</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool MatchShape(int entities, int length) => Entities == entities && Length == length;

        /// <summary>
        /// Overwrites the contents of the current instance with the input <see cref="Tensor"/>
        /// </summary>
        /// <param name="tensor">The input <see cref="Tensor"/> to copy</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe void Overwrite(in Tensor tensor)
        {
            if (tensor.Entities != Entities || tensor.Length != Length) throw new ArgumentException("The input tensor doesn't have the same size as the target");
            int bytes = sizeof(float) * Size;
            Buffer.MemoryCopy(tensor, this, bytes, bytes);
        }

        /// <summary>
        /// Overwrites the contents of the current <see cref="Tensor"/> with the input array
        /// </summary>
        /// <param name="array">The input array to copy</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe void Overwrite([NotNull] float[] array)
        {
            if (array.Length != Size) throw new ArgumentException("The input array doesn't have the same size as the target");
            int bytes = sizeof(float) * Size;
            fixed (float* p = array) Buffer.MemoryCopy(p, this, bytes, bytes);
        }

        /// <summary>
        /// Duplicates the current instance to an output <see cref="Tensor"/>
        /// </summary>
        /// <param name="tensor">The output tensor</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void Duplicate(out Tensor tensor)
        {
            New(Entities, Length, out tensor);
            tensor.Overwrite(this);
        }

        /// <summary>
        /// Copies the contents of the unmanaged array to a managed <see cref="Array"/>
        /// </summary>
        /// <param name="keepAlive">Indicates whether or not to automatically dispose the current instance</param>
        [Pure, NotNull]
        public unsafe float[] ToArray(bool keepAlive = true)
        {
            if (Ptr == IntPtr.Zero) return new float[0];
            float[] result = new float[Size];
            new Span<float>(this, Size).CopyTo(result);
            if (!keepAlive) Free();
            return result;
        }

        /// <summary>
        /// Copies the contents of the unmanaged array to a managed 2D <see cref="Array"/>
        /// </summary>
        /// <param name="keepAlive">Indicates whether or not to automatically dispose the current instance</param>
        [Pure, NotNull]
        public unsafe float[,] ToArray2D(bool keepAlive = true)
        {
            if (Ptr == IntPtr.Zero) return new float[0, 0];
            float[,] result = new float[Entities, Length];
            fixed (float* p = result)
                new Span<float>(this, Size).CopyTo(new Span<float>(p, Size));
            if (!keepAlive) Free();
            return result;
        }

        /// <summary>
        /// Returns a <see cref="Span{T}"/> representing the current <see cref="Tensor"/>
        /// </summary>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe Span<float> AsSpan() => new Span<float>(this, Size);

        #endregion

        #region Memory management

        /// <summary>
        /// Frees the memory associated with the current instance
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Free() => Marshal.FreeHGlobal(Ptr);

        /// <summary>
        /// Frees the memory associated with the current instance, if needed
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void TryFree()
        {
            if (Ptr != IntPtr.Zero)
                Marshal.FreeHGlobal(Ptr);
        }

        /// <summary>
        /// Frees the input sequence of <see cref="Tensor"/> instances
        /// </summary>
        /// <param name="tensors">The tensors to free</param>
        /// <remarks>The <see langword="params"/> usage in the method arguments will cause a heap allocation
        /// when this method is called. Manually calling <see cref="Free()"/> on each target <see cref="Tensor"/>
        /// should have a slightly better performance. The same is true for the <see cref="TryFree(Tensor[])"/> method as well.</remarks>
        public static unsafe void Free([NotNull] params Tensor[] tensors)
        {
            fixed (Tensor* p = tensors)
                for (int i = 0; i < tensors.Length; i++)
                    p[i].Free();
        }

        /// <summary>
        /// Frees the input sequence of <see cref="Tensor"/> instances, if possible
        /// </summary>
        /// <param name="tensors">The tensors to free</param>
        public static unsafe void TryFree([NotNull] params Tensor[] tensors)
        {
            fixed (Tensor* p = tensors)
                for (int i = 0; i < tensors.Length; i++)
                    p[i].TryFree();
        }

        /// <summary>
        /// Gets a raw pointer to the <see cref="Tensor"/> data
        /// </summary>
        /// <param name="tensor"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [SuppressMessage("ReSharper", "ImpureMethodCallOnReadonlyValueField")]
        public static unsafe implicit operator float*(in Tensor tensor) => (float*)tensor.Ptr.ToPointer();

        #endregion

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

            private const int MaximumRowsCount = 10;

            private const int MaximumItemsCount = 30000;

            [SuppressMessage("ReSharper", "UnusedMember.Local")]
            public _TensorProxy(Tensor obj)
            {
                IEnumerable<float[]> ExtractRows()
                {
                    unsafe float[] ExtractRow(int i)
                    {
                        float[] row = new float[obj.Length];
                        float* p = (float*)obj + obj.Length * i;
                        long bytes = sizeof(float) * row.Length;
                        fixed (float* pr = row)
                            Buffer.MemoryCopy(p, pr, bytes, bytes);
                        return row;
                    }

                    // Spawn the sequence
                    int
                        max = MaximumItemsCount / obj.Length,
                        up = max.Min(MaximumRowsCount).Max(1).Min(obj.Entities);
                    for (int i = 0; i < up; i++)
                        yield return ExtractRow(i);
                }
                RowsPreview = ExtractRows();
            }
        }

        #endregion
    }
}
