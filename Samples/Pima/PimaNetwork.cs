using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.APIs.Structs;
using System.Threading.Tasks;

namespace cux.neuro
{
    public class PimaNetwork
    {
        public INeuralNetwork Network { get; set; }

        public PimaNetwork()
        {
            Network = NetworkManager.NewSequential(TensorInfo.Linear(8),
                NetworkLayers.FullyConnected(10, ActivationType.Tanh),
                NetworkLayers.FullyConnected(10, ActivationType.Tanh),
                NetworkLayers.Softmax(2));
        }

        public async Task<TrainingSessionResult> Train(ITrainingDataset dataset, int reps = 1000)
        {
            return await NetworkManager.TrainNetworkAsync(Network,
                dataset, TrainingAlgorithms.AdaDelta(), reps, 0.5f);
        }

        public bool Predict(float[] input)
        {
            float[] output = Network.Forward(input);
            bool hasDiabetes = output[0] > 0.5;
            return hasDiabetes;
        }
    }
}
