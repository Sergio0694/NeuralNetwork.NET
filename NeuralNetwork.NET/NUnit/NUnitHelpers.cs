using System;
using JetBrains.Annotations;

namespace NeuralNetworkNET.NUnit
{
    internal static class NUnitHelpers
    {
        /// <summary>
        /// Checks if two matrices have the same size and content
        /// </summary>
        /// <param name="m">The first matrix to test</param>
        /// <param name="o">The second matrix to test</param>
        public static bool ContentEquals([CanBeNull] this double[,] m, [CanBeNull] double[,] o)
        {
            if (m == null && o == null) return true;
            if (m == null || o == null) return false;
            if (m.GetLength(0) != o.GetLength(0) ||
                m.GetLength(1) != o.GetLength(1)) return false;
            for (int i = 0; i < m.GetLength(0); i++)
            for (int j = 0; j < m.GetLength(1); j++)
                if (Math.Abs(m[i, j] - o[i, j]) > 0.0001) return false;
            return true;
        }

        /// <summary>
        /// Checks if two vectors have the same size and content
        /// </summary>
        /// <param name="v">The first vector to test</param>
        /// <param name="o">The second vector to test</param>
        public static bool ContentEquals([CanBeNull] this double[] v, [CanBeNull] double[] o)
        {
            if (v == null && o == null) return true;
            if (v == null || o == null) return false;
            if (v.Length != o.Length) return false;
            for (int i = 0; i < v.Length; i++)
                if (Math.Abs(v[i] - o[i]) > 0.0001) return false;
            return true;
        }
    }
}
