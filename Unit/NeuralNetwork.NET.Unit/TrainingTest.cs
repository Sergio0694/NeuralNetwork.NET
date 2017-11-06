using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.SupervisedLearning.Misc;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// Test class for the <see cref="SupervisedLearning.BackpropagationNetworkTrainer"/> class and dependencies
    /// </summary>
    [TestClass]
    [TestCategory(nameof(TrainingTest))]
    public class TrainingTest
    {
        [TestMethod]
        public void BatchDivisionTest1()
        {
            Random r = new Random();
            double[,]
                x = r.NextXavierMatrix(60000, 784),
                y = r.NextXavierMatrix(60000, 10);
            TrainingBatch.BatchesCollection batches = TrainingBatch.BatchesCollection.FromDataset(x, y, 1000);
            List<TrainingBatch> testList = new List<TrainingBatch>();
            for (int i = 0; i < 60; i++) testList.Add(batches.Next());
            double[,]
                xs = testList.Select(b => b.X).ToArray().MergeRows(),
                ys = testList.Select(b => b.Y).ToArray().MergeRows();
            Assert.IsTrue(x.ContentEquals(xs));
            Assert.IsTrue(y.ContentEquals(ys));
        }

        [TestMethod]
        public void BatchDivisionTest2()
        {
            Random r = new Random();
            double[,]
                x = r.NextXavierMatrix(20000, 784),
                y = r.NextXavierMatrix(20000, 10);
            TrainingBatch.BatchesCollection batches = TrainingBatch.BatchesCollection.FromDataset(x, y, 333);
            List<TrainingBatch> testList = new List<TrainingBatch>();
            for (int i = 0; i < 61; i++) testList.Add(batches.Next());
            double[,]
                xs = testList.Select(b => b.X).ToArray().MergeRows(),
                ys = testList.Select(b => b.Y).ToArray().MergeRows();
            Assert.IsTrue(x.ContentEquals(xs));
            Assert.IsTrue(y.ContentEquals(ys));
        }
    }
}
