using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.Network.Layers.Abstract.Base;

namespace NeuralNetworkDotNet.Network.Layers.Abstract
{
    /// <summary>
    /// A base <see langword="class"/> for a network layer with no parameters to optimize during training
    /// </summary>
    internal abstract class ConstantLayerBase : LayerBase
    {
        protected ConstantLayerBase(Shape input, Shape output) : base(input, output) { }

        /// <summary>
        /// Backpropagates the error to compute the delta for the inputs of the layer
        /// </summary>
        /// <param name="x">The layer inputs used in the forward pass</param>
        /// <param name="y">The output <see cref="Tensor"/> computed in the forward pass</param>
        /// <param name="dy">The output error delta to backpropagate</param>
        public abstract Tensor Backpropagate([NotNull] Tensor x, [NotNull] Tensor y, [NotNull] Tensor dy);
    }
}
