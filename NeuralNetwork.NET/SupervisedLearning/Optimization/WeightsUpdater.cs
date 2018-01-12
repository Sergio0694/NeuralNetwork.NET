using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Networks.Layers.Abstract;

namespace NeuralNetworkNET.SupervisedLearning.Optimization
{
    /// <summary>
    /// A delegate used to update the weights in a neural network after each training batch is processed
    /// </summary>
    /// <param name="i">The current offset with respect to the list of weighted layers in the network (it can be used to track external resources for each layer to update)</param>
    /// <param name="dJdw">The gradient with respect to the weights for the current layer</param>
    /// <param name="dJdb">The gradient with respect to the biases for the current layer</param>
    /// <param name="samples">The number of training samples evaluated in the current training batch</param>
    /// <param name="layer">The target layer to update</param>
    internal delegate void WeightsUpdater(int i, in Tensor dJdw, in Tensor dJdb, int samples, [NotNull] WeightedLayerBase layer);
}
