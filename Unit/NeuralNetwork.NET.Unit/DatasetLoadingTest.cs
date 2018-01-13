using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.SupervisedLearning.Data;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// Test class for the <see cref="DatasetLoader"/> class
    /// </summary>
    [TestClass]
    [TestCategory(nameof(DatasetLoadingTest))]
    public class DatasetLoadingTest
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
            HashSet<int> set1 = new HashSet<int>();
            for (int i = 0; i < 20000; i++)
            {
                set1.Add(GetUid(x, i) ^ GetUid(y, i));
            }
            HashSet<int> set2 = new HashSet<int>();
            for (int i = 0; i < batches.BatchesCount; i++)
            {
                int h = batches.Batches[i].X.GetLength(0);
                for (int j = 0; j < h; j++)
                {
                    set2.Add(GetUid(batches.Batches[i].X, j) ^ GetUid(batches.Batches[i].Y, j));
                }
            }
            Assert.IsTrue(set1.OrderBy(h => h).SequenceEqual(set2.OrderBy(h => h)));
            batches.CrossShuffle();
            HashSet<int> set3 = new HashSet<int>();
            for (int i = 0; i < batches.BatchesCount; i++)
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
            HashSet<int> set1 = new HashSet<int>();
            for (int i = 0; i < 20000; i++)
            {
                set1.Add(GetUid(x, i) ^ GetUid(y, i));
            }
            HashSet<int> set2 = new HashSet<int>();
            for (int i = 0; i < batches.BatchesCount; i++)
            {
                int h = batches.Batches[i].X.GetLength(0);
                for (int j = 0; j < h; j++)
                {
                    set2.Add(GetUid(batches.Batches[i].X, j) ^ GetUid(batches.Batches[i].Y, j));
                }
            }
            Assert.IsTrue(set1.OrderBy(h => h).SequenceEqual(set2.OrderBy(h => h)));
            batches.CrossShuffle();
            HashSet<int> set3 = new HashSet<int>();
            for (int i = 0; i < batches.BatchesCount; i++)
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
            HashSet<int> set = new HashSet<int>();
            for (int i = 0; i < 20000; i++)
            {
                set.Add(GetUid(x, i) ^ GetUid(y, i));
            }
            HashSet<int> set1 = new HashSet<int>();
            for (int i = 0; i < batches1.BatchesCount; i++)
            {
                int h = batches1.Batches[i].X.GetLength(0);
                for (int j = 0; j < h; j++)
                {
                    set1.Add(GetUid(batches1.Batches[i].X, j) ^ GetUid(batches1.Batches[i].Y, j));
                }
            }
            Assert.IsTrue(set.OrderBy(h => h).SequenceEqual(set1.OrderBy(h => h)));
            batches2.BatchSize = 1437;
            HashSet<int> set2 = new HashSet<int>();
            for (int i = 0; i < batches2.BatchesCount; i++)
            {
                int h = batches2.Batches[i].X.GetLength(0);
                for (int j = 0; j < h; j++)
                {
                    set2.Add(GetUid(batches2.Batches[i].X, j) ^ GetUid(batches2.Batches[i].Y, j));
                }
            }
            Assert.IsTrue(set.OrderBy(h => h).SequenceEqual(set2.OrderBy(h => h)));
        }

        [TestMethod]
        public void IdTest1()
        {
            // Setup
            float[,]
                x = Enumerable.Range(0, 20000 * 784).Select(_ => ThreadSafeRandom.NextUniform(100)).ToArray().AsSpan().AsMatrix(20000, 784),
                y = Enumerable.Range(0, 20000 * 10).Select(_ => ThreadSafeRandom.NextUniform(100)).ToArray().AsSpan().AsMatrix(20000, 10);
            IDataset
                set1 = DatasetLoader.Training((x, y), 1000),
                set2 = DatasetLoader.Training((x, y), 1498);
            Assert.IsTrue(set1.Id == set2.Id);
            set2.To<IDataset, BatchesCollection>().CrossShuffle();
            Assert.IsTrue(set1.Id == set2.Id);
        }

        [TestMethod]
        public void IdTest2()
        {
            // Setup
            float[,]
                x = Enumerable.Range(0, 20000 * 784).Select(_ => ThreadSafeRandom.NextUniform(100)).ToArray().AsSpan().AsMatrix(20000, 784),
                y = Enumerable.Range(0, 20000 * 10).Select(_ => ThreadSafeRandom.NextUniform(100)).ToArray().AsSpan().AsMatrix(20000, 10);
            IDataset
                set1 = DatasetLoader.Training((x, y), 1000),
                set2 = DatasetLoader.Test((x, y));
            set1.To<IDataset, BatchesCollection>().CrossShuffle();
            Assert.IsTrue(set1.Id == set2.Id);
        }

        // Calculates a unique hash code for the target vector
        private static unsafe int GetUid(float[] v)
        {
            fixed (float* pv = v)
            {
                int hash = 17;
                unchecked
                {
                    for (int i = 0; i < v.Length; i++)
                        hash = hash * 23 + pv[i].GetHashCode();
                    return hash;
                }
            }
        }

        [TestMethod]
        public void DatasetPartition()
        {
            float[,]
                x = Enumerable.Range(0, 20000 * 784).Select(_ => ThreadSafeRandom.NextUniform(100)).ToArray().AsSpan().AsMatrix(20000, 784),
                y = Enumerable.Range(0, 20000 * 10).Select(_ => ThreadSafeRandom.NextUniform(100)).ToArray().AsSpan().AsMatrix(20000, 10);
            ITrainingDataset sourceDataset = DatasetLoader.Training((x, y), 1000);
            (ITrainingDataset training, ITestDataset test) = sourceDataset.PartitionWithTest(0.33f);
            HashSet<int> set = new HashSet<int>();
            for (int i = 0; i < 20000; i++)
            {
                set.Add(GetUid(x, i) ^ GetUid(y, i));
            }
            HashSet<int> set1 = new HashSet<int>();
            for (int i = 0; i < training.Count; i++)
            {
                DatasetSample sample = training[i];
                set1.Add(GetUid(sample.X.ToArray()) ^ GetUid(sample.Y.ToArray()));
            }
            for (int i = 0; i < test.Count; i++)
            {
                DatasetSample sample = test[i];
                set1.Add(GetUid(sample.X.ToArray()) ^ GetUid(sample.Y.ToArray()));
            }
            Assert.IsTrue(set.OrderBy(h => h).SequenceEqual(set1.OrderBy(h => h)));
        }

        [TestMethod]
        public void DatasetPartitionException()
        {
            float[,]
                x = Enumerable.Range(0, 15 * 784).Select(_ => ThreadSafeRandom.NextUniform(100)).ToArray().AsSpan().AsMatrix(15, 784),
                y = Enumerable.Range(0, 15 * 10).Select(_ => ThreadSafeRandom.NextUniform(100)).ToArray().AsSpan().AsMatrix(15, 10);
            ITrainingDataset sourceDataset = DatasetLoader.Training((x, y), 1000);
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => sourceDataset.PartitionWithTest(0.33f));
        }
    }
}
