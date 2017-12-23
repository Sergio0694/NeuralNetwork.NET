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
    [DebuggerTypeProxy(typeof(_TensorProxy))]
    [DebuggerDisplay("Entities: {Entities}, Length: {Length}, Ptr: {Ptr}")]
    public readonly struct Tensor
    {
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
        public int Size => Entities * Length;

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
        /// <param name="n">The height of the matrix</param>
        /// <param name="chw">The width of the matrix</param>
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
        /// <param name="n">The height of the matrix</param>
        /// <param name="chw">The width of the matrix</param>
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
        /// <param name="n">The height of the final matrix</param>
        /// <param name="chw">The width of the final matrix</param>
        /// <param name="tensor">The resulting instance</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void Reshape(float* p, int n, int chw, out Tensor tensor)
        {
            IntPtr ptr = new IntPtr(p);
            tensor = new Tensor(ptr, n, chw);
        }

        /// <summary>
        /// Creates a new instance by copying the contents at the given memory location and reshaping it to the desired size
        /// </summary>
        /// <param name="p">The target memory area to copy</param>
        /// <param name="n">The height of the final matrix</param>
        /// <param name="chw">The width of the final matrix</param>
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
        /// <param name="n">The height of the final matrix</param>
        /// <param name="chw">The width of the final matrix</param>
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
        /// Overwrites the contents of the current matrix with the input matrix
        /// </summary>
        /// <param name="source">The input tensor to copy</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe void Overwrite(in Tensor tensor)
        {
            if (tensor.Entities != Entities || tensor.Length != Length) throw new ArgumentException("The input matrix doesn't have the same size as the target");
            int bytes = sizeof(float) * Size;
            Buffer.MemoryCopy(tensor, this, bytes, bytes);
        }

        /// <summary>
        /// Duplicates the current instance to an output <see cref="Tensor"/> matrix
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
        [Pure, NotNull]
        public float[] ToArray()
        {
            if (Ptr == IntPtr.Zero) return new float[0];
            float[] result = new float[Size];
            Marshal.Copy(Ptr, result, 0, Size);
            return result;
        }

        /// <summary>
        /// Copies the contents of the unmanaged array to a managed 2D <see cref="Array"/>
        /// </summary>
        [Pure, NotNull]
        public unsafe float[,] ToArray2D()
        {
            if (Ptr == IntPtr.Zero) return new float[0, 0];
            float[,] result = new float[Entities, Length];
            int size = sizeof(float) * Size;
            fixed (float* presult = result)
                Buffer.MemoryCopy(this, presult, size, size);
            return result;
        }

        #endregion

        /// <summary>
        /// Frees the memory associated with the current instance
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Free() => Marshal.FreeHGlobal(Ptr);

        // Implicit pointer conversion
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe implicit operator float*(in Tensor tensor) => (float*)tensor.Ptr.ToPointer();

        #region Debug

        /// <summary>
        /// A proxy type to debug instances of the <see cref="Tensor"/> <see cref="struct"/>
        /// </summary>
        private struct _TensorProxy
        {
            /// <summary>
            /// Gets a preview of the underlying memory area wrapped by this instance
            /// </summary>
            [NotNull]
            [SuppressMessage("ReSharper", "MemberCanBePrivate.Local")]
            [SuppressMessage("ReSharper", "UnusedAutoPropertyAccessor.Local")]
            public IEnumerable<float[]> RowsPreview { get; }

            private const int MaximumRowsCount = 10;

            private const int MaximumItemsCount = 40000;

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
                        up = max.Min(MaximumRowsCount).Max(1);
                    for (int i = 0; i < up; i++)
                        yield return ExtractRow(i);
                }
                RowsPreview = ExtractRows();
            }
        }

        #endregion
    }
}
