using System;
using NeuralNetworkLibrary.Networks.Implementations;

namespace NeuralNetworkLibrary.Networks.PublicAPIs
{
    /// <summary>
    /// A static class that can deserialize a neural network from a byte array
    /// </summary>
    public static class NeuralNetworkDeserializer
    {
        /// <summary>
        /// Tries to deserialize the neural network
        /// </summary>
        /// <param name="data">The input serialization data</param>
        public static INeuralNetwork TryGetInstance(byte[] data)
        {
            try
            {
                // Get the int parameters
                int position = 0,
                    input = BitConverter.ToInt32(data, position),
                    hidden = BitConverter.ToInt32(data, position += 4),
                    output = BitConverter.ToInt32(data, position += 4),
                    w2w = BitConverter.ToInt32(data, position += 4);

                // Get the thresholds
                double
                    z1Th = BitConverter.ToDouble(data, position += 4),
                    z2Th = BitConverter.ToDouble(data, position += 8);
                double? z1nTh, z2nTh;
                if (z1Th == double.MinValue) z1nTh = null;
                else z1nTh = z1Th;
                if (z2Th == double.MinValue) z2nTh = null;
                else z2nTh = z2Th;

                // Deserialize the two weights matrices
                double[,] w1 = new double[input, hidden], w2 = new double[hidden, w2w];
                for (int i = 0; i < input; i++)
                {
                    for (int j = 0; j < hidden; j++)
                    {
                        w1[i, j] = BitConverter.ToDouble(data, position += 8);
                    }
                }
                for (int i = 0; i < hidden; i++)
                {
                    for (int j = 0; j < w2w; j++)
                    {
                        w1[i, j] = BitConverter.ToDouble(data, position += 8);
                    }
                }
                position += 8;

                // Check if the network has two layers
                if (data.Length > position)
                {
                    // Get the new parameters
                    int second = BitConverter.ToInt32(data, position);
                    double z3Th = BitConverter.ToDouble(data, position += 4);
                    double? z3nTh;
                    if (z3Th == double.MinValue) z3nTh = null;
                    else z3nTh = z3Th;
                    double[,] w3 = new double[second, output];
                    for (int i = 0; i < second; i++)
                    {
                        for (int j = 0; j < output; j++)
                        {
                            w3[i, j] = BitConverter.ToDouble(data, position += 8);
                        }
                    }

                    // Return the two layers network
                    return new TwoLayersNeuralNetwork(input, output, hidden, second, w1, w2, w3, z1nTh, z2nTh, z3nTh);
                }
                return new NeuralNetwork(input, output, hidden, w1, w2, z1nTh, z2nTh);
            }
            catch
            {
                // Whops!
                return null;
            }
        }
    }
}
