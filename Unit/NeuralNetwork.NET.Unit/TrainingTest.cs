using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.SupervisedLearning;
using NeuralNetworkNET.SupervisedLearning.Misc;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// Test class for the <see cref="NetworkTrainer"/> class and dependencies
    /// </summary>
    [TestClass]
    [TestCategory(nameof(TrainingTest))]
    public class TrainingTest
    {
        [TestMethod]
        public void BatchDivisionTest1()
        {
            Random r = new Random();
            float[,]
                x = r.NextXavierMatrix(60000, 784),
                y = r.NextXavierMatrix(60000, 10);
            BatchesCollection batches = BatchesCollection.FromDataset((x, y), 1000);
            IEnumerable<TrainingBatch> testList = batches.NextEpoch();
            // TODO: check the shuffle is coherent
        }

        [TestMethod]
        public void BatchDivisionTest2()
        {
            Random r = new Random();
            float[,]
                x = r.NextXavierMatrix(20000, 784),
                y = r.NextXavierMatrix(20000, 10);
            BatchesCollection batches = BatchesCollection.FromDataset((x, y), 333);
            IEnumerable<TrainingBatch> testList = batches.NextEpoch();
            // TODO: check the shuffle is coherent
        }
    }
}
