using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Networks.Activations;

namespace NeuralNetworkNET.Networks.Layers.Abstract
{
    /// <summary>
    /// A base class for a network layer with no parameters to optimize during training
    /// </summary>
    internal abstract class ConstantLayerBase : NetworkLayerBase
    {
        protected ConstantLayerBase(in TensorInfo input, in TensorInfo output, ActivationFunctionType activation) 
            : base(in input, in output, activation) { }

        /// <summary>
        /// Backpropagates the error to compute the delta for the inputs of the layer
        /// </summary>
        /// <param name="z">The output <see cref="Tensor"/> computed in the forward pass</param>
        /// <param name="a">The layer activation computed in the forward pass</param>
        /// <param name="dy">The output error delta to backpropagate</param>
        /// <param name="x">The layer inputs used in the forward pass</param>
        /// <param name="dx">The resulting backpropagated error</param>
        public abstract void Backpropagate(in Tensor z, in Tensor a, in Tensor dy, in Tensor x, in Tensor dx);
    }
}
