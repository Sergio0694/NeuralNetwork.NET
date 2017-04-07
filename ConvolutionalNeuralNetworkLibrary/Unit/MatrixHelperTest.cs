using System;
using NUnit.Framework;

namespace ConvolutionalNeuralNetworkLibrary.Unit
{
    /// <summary>
    /// Test class for the <see cref="MatrixHelper"/> class
    /// </summary>
    [TestFixture]
    [Category(nameof(MatrixHelper))]
    internal static class MatrixHelperTest
    {
        #region CNN

        /// <summary>
        /// ReLU test
        /// </summary>
        [Test]
        [Category(nameof(MatrixHelper))]
        [Category("CNN")]
        public static void ReLU()
        {
            // Test values
            double[,]
                m =
                {
                    { -1, -0.1, 2 },
                    { 1, 1, 2 },
                    { 0, -0.3, 99 }
                },
                r =
                {
                    { 0, 0, 2 },
                    { 1, 1, 2 },
                    { 0, 0, 99 }
                },
                t = m.ReLU();
            Assert.True(t.ContentEquals(r));
        }

        /// <summary>
        /// Pool 2x2 test
        /// </summary>
        [Test]
        [Category(nameof(MatrixHelper))]
        [Category("CNN")]
        public static void Pool2x2()
        {
            // Test values
            double[,]
                m =
                {
                    { -1, 0, 1, 2 },
                    { 1, 1, 1, 1 },
                    { 0, -0.3, -5, -0.5 },
                    { -1, 10, -2, -1 }
                },
                r =
                {
                    { 1, 2 },
                    { 10, -0.5 }
                },
                t = m.Pool2x2();
            Assert.True(t.ContentEquals(r));
        }

        #endregion

        #region Misc

        /// <summary>
        /// Vector-matrix multiplication test
        /// </summary>
        [Test]
        [Category("Misc")]
        public static void LinearMultiplication()
        {
            // Test values
            double[,] m =
            {
                { 1, 1, 1, 1 },
                { 0, 2, -1, 0 },
                { 1, 1, 1, 1 },
                { 0, 0, -1, 1 }
            };
            double[]
                v = { 1, 2, 0.1, -2 },
                r = { 1.1, 5.1, 1.1, -0.9 },
                t = v.Multiply(m);
            Assert.True(t.ContentEquals(r));

            // Exception test
            double[] f = { 1, 2, 3, 4, 5, 6 };
            Assert.Throws<ArgumentOutOfRangeException>(() => f.Multiply(m));
        }

        /// <summary>
        /// Matrix-matrix multiplication test
        /// </summary>
        [Test]
        [Category("Misc")]
        public static void SpatialMultiplication()
        {
            // Test values
            double[,]
                m1 =
                {
                    { 1, 2, 3 },
                    { 5, 0.1, -2 }
                },
                m2 =
                {
                    { 5, 2, -1, 3 },
                    { -5, 2, -7, 0.9 },
                    { 0.1, 0.2, -0.1, 2 }
                },
                r =
                {
                    { -4.7, 6.6, -15.3, 10.8 },
                    { 24.3, 9.7999, -5.5, 11.09 }
                },
                t = m1.Multiply(m2);
            Assert.True(t.ContentEquals(r));

            // Exception test
            double[,] f =
                {
                    { 1, 2, 1, 0, 0 },
                    { 5, 0.1, 0, 0, 0 }
                };
            Assert.Throws<ArgumentOutOfRangeException>(() => f.Multiply(m1));
            Assert.Throws<ArgumentOutOfRangeException>(() => m2.Multiply(f));
        }

        /// <summary>
        /// Matrix transposition
        /// </summary>
        [Test]
        [Category("Misc")]
        public static void Transposition()
        {
            // Test values
            double[,]
                m =
                {
                    { 1, 1, 1, 1 },
                    { 0, 2, -1, 0 }
                },
                r =
                {
                    { 1, 0 },
                    { 1, 2 },
                    { 1, -1 },
                    { 1, 0 }
                },
                t = m.Transpose();
            Assert.True(t.ContentEquals(r));
        }

        #endregion

        #region Helpers

        // Helper method for matrices
        private static bool ContentEquals(this double[,] m, double[,] o)
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

        // Helper method for vectors
        private static bool ContentEquals(this double[] v, double[] o)
        {
            if (v == null && o == null) return true;
            if (v == null || o == null) return false;
            if (v.Length != o.Length) return false;
            for (int i = 0; i < v.Length; i++)
                if (Math.Abs(v[i] - o[i]) > 0.0001) return false;
            return true;
        }

        #endregion
    }
}
