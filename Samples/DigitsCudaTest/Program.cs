using System;
using System.Diagnostics;
using Alea;
using Alea.Parallel;
using NeuralNetworkNET.Helpers;

namespace DigitsCudaTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Random random = new Random(DateTime.Now.Millisecond);
            int x = 2000, y = 1500, z = 800;
            double[,]
                a = random.NextMatrix(x, y),
                b = random.NextMatrix(y, z);

            Stopwatch timer = new Stopwatch();
            timer.Start();
            var c1 = new double[x, z];
            int m = a.GetLength(0);
            int n = a.GetLength(1);
            int p = b.GetLength(1);

            Gpu.Default.For(0, m * p, index =>
            {
                int
                    i = index / p,
                    j = index % p;

                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += a[i, k] * b[k, j];
                }
                c1[i, j] = sum;
            });

            timer.Stop();
            var t1 = timer.Elapsed;
            timer.Restart();
            var c2 = a.Multiply(b);
            timer.Stop();
            var t2 = timer.Elapsed;
            Debug.Assert(c1.ContentEquals(c2));
            Debug.WriteLine($"{t1.TotalMilliseconds} vs {t2.TotalMilliseconds}");
        }
    }
}
