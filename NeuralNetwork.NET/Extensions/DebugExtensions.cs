using System;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Structs;

namespace NeuralNetworkNET.Extensions
{
    /// <summary>
    /// A static class that contains some extensions to check the contents of various types of <see cref="float"/> vectors
    /// </summary>
    internal static class DebugExtensions
    {
        /// <summary>
        /// Checks if two <see cref="Span{T}"/> instances have the same size and content
        /// </summary>
        /// <param name="x1">The first <see cref="Span{T}"/> to test</param>
        /// <param name="x2">The second <see cref="Span{T}"/> to test</param>
        /// <param name="absolute">The relative comparison threshold</param>
        /// <param name="relative">The relative comparison threshold</param>
        public static unsafe bool ContentEquals(this Span<float> x1, Span<float> x2, float absolute = 1e-6f, float relative = 1e-6f)
        {
            if (x1.Length != x2.Length) return false;
            fixed (float* p1 = x1, p2 = x2)
            {
                for (int i = 0; i < x1.Length; i++)
                    if (!p1[i].EqualsWithDelta(p2[i], absolute, relative))
                    {
                        #if DEBUG
                        System.Diagnostics.Debug.WriteLine($"[NO MATCH] {p1[i]} | {p2[i]} | diff: {(p1[i] - p2[i]).Abs()}");
                        #endif
                        return false;
                    }
            }
            return true;
        }

        /// <summary>
        /// Checks if two <see cref="Tensor"/> instances have the same size and content
        /// </summary>
        /// <param name="m">The first <see cref="Tensor"/> to test</param>
        /// <param name="o">The second <see cref="Tensor"/> to test</param>
        /// <param name="absolute">The relative comparison threshold</param>
        /// <param name="relative">The relative comparison threshold</param>
        public static bool ContentEquals(in this Tensor m, in Tensor o, float absolute = 1e-6f, float relative = 1e-6f)
        {
            if (m.Ptr == IntPtr.Zero && o.Ptr == IntPtr.Zero) return true;
            if (m.Ptr == IntPtr.Zero || o.Ptr == IntPtr.Zero) return false;
            if (m.Entities != o.Entities || m.Length != o.Length) return false;
            return m.AsSpan().ContentEquals(o.AsSpan(), absolute, relative);
        }

        /// <summary>
        /// Checks if two matrices have the same size and content
        /// </summary>
        /// <param name="m">The first matrix to test</param>
        /// <param name="o">The second matrix to test</param>
        /// <param name="absolute">The relative comparison threshold</param>
        /// <param name="relative">The relative comparison threshold</param>
        public static unsafe bool ContentEquals([CanBeNull] this float[,] m, [CanBeNull] float[,] o, float absolute = 1e-6f, float relative = 1e-6f)
        {
            if (m == null && o == null) return true;
            if (m == null || o == null) return false;
            if (m.GetLength(0) != o.GetLength(0) ||
                m.GetLength(1) != o.GetLength(1)) return false;
            fixed (float* pm = m, po = o)
                return new Span<float>(pm, m.Length).ContentEquals(new Span<float>(po, o.Length), absolute, relative);
        }

        /// <summary>
        /// Checks if two vectors have the same size and content
        /// </summary>
        /// <param name="v">The first vector to test</param>
        /// <param name="o">The second vector to test</param>
        /// <param name="absolute">The relative comparison threshold</param>
        /// <param name="relative">The relative comparison threshold</param>
        public static bool ContentEquals([CanBeNull] this float[] v, [CanBeNull] float[] o, float absolute = 1e-6f, float relative = 1e-6f)
        {
            if (v == null && o == null) return true;
            if (v == null || o == null) return false;
            return v.AsSpan().ContentEquals(o, absolute, relative);
        }
    }
}
