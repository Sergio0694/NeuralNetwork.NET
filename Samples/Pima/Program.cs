using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace cux.neuro
{
    class Program
    {
        static void Main(string[] args)
        {
            // https://www.kaggle.com/uciml/pima-indians-diabetes-database
            var data = PimaUtil.LoadFromFile(@"diabetes.csv");
            var testData = data.Take(80);
            var trainingData = data.Skip(80);

            ITrainingDataset dataset = DatasetLoader.Training(trainingData, 100);

            var net = new PimaNetwork();
            net.Train(dataset).Wait();

            Console.Write("Accuracy at learning: ");
            float learningRatio = GetValidRatio(trainingData, net);
            Console.WriteLine($"{learningRatio:P1}");

            Console.Write("Accuracy at test: ");
            float testRatio = GetValidRatio(testData, net);
            Console.WriteLine($"{testRatio:P1}");
        }

        // get valid to all ratio
        private static float GetValidRatio(IEnumerable<(float[] x, float[] y)> data, PimaNetwork net)
        {
            (float[] x, float[] u)[] dataAsArray = data.ToArray();
            int validCases = 0;
            for (int i = 0; i < dataAsArray.Length; i++)
            {
                var (input, output) = dataAsArray[i];
                bool hasDiabetesPred = net.Predict(input);
                bool hasDiabetesAct = output[0] > 0.5;
                if (hasDiabetesPred == hasDiabetesAct)
                {
                    validCases++;
                }
            }
            return validCases / (float)dataAsArray.Length;
        }
    }
}
