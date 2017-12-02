using JetBrains.Annotations;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace NeuralNetworkNET.Structs
{
    /// <summary>
    /// A readonly struct that holds the info on an unmanaged 2D memory area that has been allocated
    /// </summary>
    [DebuggerDisplay("Height: {Height}, Width: {Width}, Ptr: {Ptr}")]
    public readonly ref struct FloatSpan2D
    {
        /// <summary>
        /// Gets the <see cref="IntPtr"/> value to the allocated memory
        /// </summary>
        public readonly IntPtr Ptr;

        /// <summary>
        /// Gets the height of the memory area
        /// </summary>
        public readonly int Height;

        /// <summary>
        /// Gets the width of the memory area
        /// </summary>
        public readonly int Width;

        /// <summary>
        /// Gets the total number of items in the allocated matrix
        /// </summary>
        public int Size => Height * Width;

        #region Initialization

        // Private constructor
        private FloatSpan2D(IntPtr ptr, int height, int width)
        {
            Ptr = ptr;
            Height = height;
            Width = width;
        }

        /// <summary>
        /// Creates a new instance with the specified shape
        /// </summary>
        /// <param name="height">The height of the matrix</param>
        /// <param name="width">The width of the matrix</param>
        /// <param name="span">The resulting instance</param>
        public static void New(int height, int width, out FloatSpan2D span)
        {
            IntPtr ptr = Marshal.AllocHGlobal(sizeof(float) * height * width);
            span = new FloatSpan2D(ptr, height, width);
        }

        /// <summary>
        /// Creates a new instance by wrapping the input pointer
        /// </summary>
        /// <param name="p">The target memory area</param>
        /// <param name="height">The height of the final matrix</param>
        /// <param name="width">The width of the final matrix</param>
        /// <param name="span">The resulting instance</param>
        public static unsafe void Fix(float* p, int height, int width, out FloatSpan2D span)
        {
            IntPtr ptr = new IntPtr(p);
            span = new FloatSpan2D(ptr, height, width);
        }

        /// <summary>
        /// Creates a new instance by copying the contents at the given memory location and reshaping it to the desired size
        /// </summary>
        /// <param name="p">The target memory area to copy</param>
        /// <param name="height">The height of the final matrix</param>
        /// <param name="width">The width of the final matrix</param>
        /// <param name="span">The resulting instance</param>
        public static unsafe void From(float* p, int height, int width, out FloatSpan2D span)
        {
            New(height, width, out span);
            int size = sizeof(float) * height * width;
            Buffer.MemoryCopy(p, span, size, size);
        }

        /// <summary>
        /// Creates a new instance by copying the contents of the input matrix
        /// </summary>
        /// <param name="m">The input matrix to copy</param>
        /// <param name="span">The resulting instance</param>
        public static unsafe void From([NotNull] float[,] m, out FloatSpan2D span)
        {
            fixed (float* pm = m)
                From(pm, m.GetLength(0), m.GetLength(1), out span);
        }

        /// <summary>
        /// Creates a new instance by copying the contents of the input vector and reshaping it to the desired size
        /// </summary>
        /// <param name="v">The input vector to copy</param>
        /// <param name="height">The height of the final matrix</param>
        /// <param name="width">The width of the final matrix</param>
        /// <param name="span">The resulting instance</param>
        public static unsafe void From([NotNull] float[] v, int height, int width, out FloatSpan2D span)
        {
            if (height * width != v.Length) throw new ArgumentOutOfRangeException(nameof(v), "The input vector doesn't have a valid size");
            fixed (float* pv = v)
                From(pv, height, width, out span);
        }

        #endregion

        /// <summary>
        /// Overwrites the contents of the current matrix with the input matrix
        /// </summary>
        /// <param name="source">The input matrix to copy</param>
        public unsafe void Overwrite(in FloatSpan2D source)
        {
            if (source.Height != Height || source.Width != Width) throw new ArgumentException("The input matrix doesn't have the same size as the target");
            int bytes = sizeof(float) * Size;
            Buffer.MemoryCopy(source, this, bytes, bytes);
        }

        /// <summary>
        /// Copies the contents of the unmanaged array to a managed <see cref="Array"/>
        /// </summary>
        [Pure, NotNull]
        public float[] ToArray()
        {
            float[] result = new float[Height * Width];
            Marshal.Copy(Ptr, result, 0, Size);
            return result;
        }

        /// <summary>
        /// Copies the contents of the unmanaged array to a managed 2D <see cref="Array"/>
        /// </summary>
        [Pure, NotNull]
        public unsafe float[,] ToArray2D()
        {
            float[,] result = new float[Height, Width];
            int size = sizeof(float) * Size;
            fixed (float* presult = result)
                Buffer.MemoryCopy(this, presult, size, size);
            return result;
        }

        /// <summary>
        /// Frees the memory associated with the current instance
        /// </summary>
        public void Free() => Marshal.FreeHGlobal(Ptr);

        // Implicit pointer conversion
        public static unsafe implicit operator float* (in FloatSpan2D span) => (float*)span.Ptr.ToPointer();
    }
}
