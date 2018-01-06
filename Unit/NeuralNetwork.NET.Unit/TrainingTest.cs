using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.SupervisedLearning.Data;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// Test class for the <see cref="Networks.Implementations.NetworkTrainer"/> class and dependencies
    /// </summary>
    [TestClass]
    [TestCategory(nameof(TrainingTest))]
    public class TrainingTest
    {
        // Calculates a unique hash code for the target row of the input matrix
        private static unsafe int GetUid(float[,] m, int row)
        {
            int
                w = m.GetLength(1),
                offset = row * w;
            fixed (float* pm = m)
            {
                float* p = pm + offset;
                int hash = 17;
                unchecked
                {
                    for (int i = 0; i < w; i++)
                        hash = hash * 23 + p[i].GetHashCode();
                    return hash;
                }
            }
        }

        [TestMethod]
        public void BatchDivisionTest1()
        {
            // Sequential
            float[,]
                x = Enumerable.Range(0, 20000 * 784).Select(_ => ThreadSafeRandom.NextUniform(100)).ToArray().AsSpan().AsMatrix(20000, 784),
                y = Enumerable.Range(0, 20000 * 10).Select(_ => ThreadSafeRandom.NextUniform(100)).ToArray().AsSpan().AsMatrix(20000, 10);
            BatchesCollection batches = BatchesCollection.From((x, y), 1000);
            HashSet<int>
                set1 = new HashSet<int>();
            for (int i = 0; i < 20000; i++)
            {
                set1.Add(GetUid(x, i) ^ GetUid(y, i));
            }
            HashSet<int>
                set2 = new HashSet<int>();
            for (int i = 0; i < batches.BatchSize; i++)
            {
                int h = batches.Batches[i].X.GetLength(0);
                for (int j = 0; j < h; j++)
                {
                    set2.Add(GetUid(batches.Batches[i].X, j) ^ GetUid(batches.Batches[i].Y, j));
                }
            }
            Assert.IsTrue(set1.OrderBy(h => h).SequenceEqual(set2.OrderBy(h => h)));
            batches.CrossShuffle();
            HashSet<int>
                set3 = new HashSet<int>();
            for (int i = 0; i < batches.BatchSize; i++)
            {
                int h = batches.Batches[i].X.GetLength(0);
                for (int j = 0; j < h; j++)
                {
                    set3.Add(GetUid(batches.Batches[i].X, j) ^ GetUid(batches.Batches[i].Y, j));
                }
            }
            Assert.IsTrue(set1.OrderBy(h => h).SequenceEqual(set3.OrderBy(h => h)));
        }

        [TestMethod]
        public void BatchDivisionTest2()
        {
            // Sequential
            float[,]
                x = Enumerable.Range(0, 20000 * 784).Select(_ => ThreadSafeRandom.NextUniform(100)).ToArray().AsSpan().AsMatrix(20000, 784),
                y = Enumerable.Range(0, 20000 * 10).Select(_ => ThreadSafeRandom.NextUniform(100)).ToArray().AsSpan().AsMatrix(20000, 10);
            BatchesCollection batches = BatchesCollection.From((x, y), 1547);
            HashSet<int>
                set1 = new HashSet<int>();
            for (int i = 0; i < 20000; i++)
            {
                set1.Add(GetUid(x, i) ^ GetUid(y, i));
            }
            HashSet<int>
                set2 = new HashSet<int>();
            for (int i = 0; i < batches.BatchSize; i++)
            {
                int h = batches.Batches[i].X.GetLength(0);
                for (int j = 0; j < h; j++)
                {
                    set2.Add(GetUid(batches.Batches[i].X, j) ^ GetUid(batches.Batches[i].Y, j));
                }
            }
            Assert.IsTrue(set1.OrderBy(h => h).SequenceEqual(set2.OrderBy(h => h)));
            batches.CrossShuffle();
            HashSet<int>
                set3 = new HashSet<int>();
            for (int i = 0; i < batches.BatchSize; i++)
            {
                int h = batches.Batches[i].X.GetLength(0);
                for (int j = 0; j < h; j++)
                {
                    set3.Add(GetUid(batches.Batches[i].X, j) ^ GetUid(batches.Batches[i].Y, j));
                }
            }
            Assert.IsTrue(set1.OrderBy(h => h).SequenceEqual(set3.OrderBy(h => h)));
        }

        [TestMethod]
        public void BatchInitializationTest()
        {
            float[,]
                x = Enumerable.Range(0, 250 * 600).Select(_ => ThreadSafeRandom.NextUniform(1000)).ToArray().AsSpan().AsMatrix(250, 600),
                y = Enumerable.Range(0, 250 * 10).Select(_ => ThreadSafeRandom.NextUniform(500)).ToArray().AsSpan().AsMatrix(250, 10);
            (float[], float[])[] samples = Enumerable.Range(0, 250).Select(i =>
            {
                float[]
                    xv = new float[600],
                    yv = new float[10];
                Buffer.BlockCopy(x, sizeof(float) * i * 600, xv, 0, sizeof(float) * 600);
                Buffer.BlockCopy(y, sizeof(float) * i * 10, yv, 0, sizeof(float) * 10);
                return (xv, yv);
            }).ToArray();
            BatchesCollection
                batch1 = BatchesCollection.From((x, y), 100),
                batch2 = BatchesCollection.From(samples, 100);
            Assert.IsTrue(batch1.Batches.Zip(batch2.Batches, (b1, b2) => b1.X.ContentEquals(b2.X) && b1.Y.ContentEquals(b2.Y)).All(b => b));
        }

        [TestMethod]
        public void ReshapeTest()
        {
            // Setup
            float[,]
                x = Enumerable.Range(0, 20000 * 784).Select(_ => ThreadSafeRandom.NextUniform(100)).ToArray().AsSpan().AsMatrix(20000, 784),
                y = Enumerable.Range(0, 20000 * 10).Select(_ => ThreadSafeRandom.NextUniform(100)).ToArray().AsSpan().AsMatrix(20000, 10);
            BatchesCollection 
                batches1 = BatchesCollection.From((x, y), 1000),
                batches2 = BatchesCollection.From((x, y), 1000);
            HashSet<int>
                set = new HashSet<int>();
            for (int i = 0; i < 20000; i++)
            {
                set.Add(GetUid(x, i) ^ GetUid(y, i));
            }
            HashSet<int>
                set1 = new HashSet<int>();
            for (int i = 0; i < batches1.BatchSize; i++)
            {
                int h = batches1.Batches[i].X.GetLength(0);
                for (int j = 0; j < h; j++)
                {
                    set1.Add(GetUid(batches1.Batches[i].X, j) ^ GetUid(batches1.Batches[i].Y, j));
                }
            }
            Assert.IsTrue(set.OrderBy(h => h).SequenceEqual(set1.OrderBy(h => h)));
            batches2.BatchSize = 1437;
            HashSet<int>
                set2 = new HashSet<int>();
            for (int i = 0; i < batches2.BatchSize; i++)
            {
                int h = batches2.Batches[i].X.GetLength(0);
                for (int j = 0; j < h; j++)
                {
                    set2.Add(GetUid(batches2.Batches[i].X, j) ^ GetUid(batches2.Batches[i].Y, j));
                }
            }
            Assert.IsTrue(set.OrderBy(h => h).SequenceEqual(set2.OrderBy(h => h)));
        }
    }
}
