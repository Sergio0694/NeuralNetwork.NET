using JetBrains.Annotations;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace NeuralNetworkNET.Structs
{
    /// <summary>
    /// A readonly struct that holds the info on an unmanaged memory area that has been allocated
    /// </summary>
    [DebuggerDisplay("Length: {Length}, Ptr: {Ptr}")]
    public readonly ref struct FloatSpan
    {
        /// <summary>
        /// Gets the <see cref="IntPtr"/> value to the allocated memory
        /// </summary>
        public readonly IntPtr Ptr;

        /// <summary>
        /// Gets the length of the memory area
        /// </summary>
        public readonly int Length;

        #region Initialization

        // Private constructor
        private FloatSpan(IntPtr ptr, int length)
        {
            Ptr = ptr;
            Length = length;
        }

        /// <summary>
        /// Creates a new instance with a reference to the memory area
        /// </summary>
        /// <param name="p">The target pointer to wrap</param>
        /// <param name="length">The length of the vector</param>
        /// <param name="span">The resulting instance</param>
        public static unsafe void Fix(float* p, int length, out FloatSpan span)
        {
            IntPtr ptr = new IntPtr(p);
            span = new FloatSpan(ptr, length);
        }

        /// <summary>
        /// Creates a new instance with the specified length
        /// </summary>
        /// <param name="length">The length of the vector</param>
        /// <param name="span">The resulting instance</param>
        public static void New(int length, out FloatSpan span)
        {
            IntPtr ptr = Marshal.AllocHGlobal(sizeof(float) * length);
            span = new FloatSpan(ptr, length);
        }

        /// <summary>
        /// Creates a new instance by copying the contents of the input vector
        /// </summary>
        /// <param name="v">The input vector to copy</param>
        /// <param name="span">The resulting instance</param>
        public static unsafe void From([NotNull] float[] v, out FloatSpan span)
        {
            New(v.Length, out span);
            int size = sizeof(float) * v.Length;
            fixed (float* pv = v)
                Buffer.MemoryCopy(pv, span, size, size);
        }

        #endregion

        /// <summary>
        /// Copies the contents of the unmanaged array to a managed <see cref="Array"/>
        /// </summary>
        [Pure, NotNull]
        public float[] ToArray()
        {
            float[] result = new float[Length];
            Marshal.Copy(Ptr, result, 0, Length);
            return result;
        }

        /// <summary>
        /// Frees the memory associated with the current instance
        /// </summary>
        public void Free() => Marshal.FreeHGlobal(Ptr);

        // Implicit pointer conversion
        public static unsafe implicit operator float*(in FloatSpan span) => (float*)span.Ptr.ToPointer();
    }
}
