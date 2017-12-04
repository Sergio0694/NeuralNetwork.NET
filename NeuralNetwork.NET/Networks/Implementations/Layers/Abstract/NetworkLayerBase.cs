using System;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using NeuralNetworkNET.Structs;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Implementations.Layers.Abstract
{
    /// <summary>
    /// The base class for all the neural network layer implementations
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    internal abstract class NetworkLayerBase : INetworkLayer
    {
        #region Parameters

        /// <inheritdoc/>
        [JsonProperty(nameof(LayerType), Required = Required.Always, Order = 1)]
        public LayerType LayerType
        {
            get
            {
                Enum.TryParse(GetType().Name.Replace("Layer", String.Empty), out LayerType type);
                return type;
            }
        }

        /// <inheritdoc/>
        [JsonProperty(nameof(Inputs), Required = Required.Always, Order = 2)]
        public abstract int Inputs { get; }

        /// <inheritdoc/>
        [JsonProperty(nameof(Outputs), Required = Required.Always, Order = 3)]
        public abstract int Outputs { get; }

        /// <summary>
        /// Gets the activation type used in the current layer
        /// </summary>
        [JsonProperty(nameof(ActivationFunctionType), Required = Required.Always, Order = 8)]
        public ActivationFunctionType ActivationFunctionType { get; }

        /// <summary>
        /// Gets the list of activation and activation prime functions used in the network
        /// </summary>
        public (ActivationFunction Activation, ActivationFunction ActivationPrime) ActivationFunctions { get; }

        #endregion

        protected NetworkLayerBase(ActivationFunctionType activation)
        {
            ActivationFunctionType = activation;
            ActivationFunctions = ActivationFunctionProvider.GetActivations(activation);
        }

        /// <summary>
        /// Forwards the inputs through the network layer and returns the resulting activity (Z) and activation (A)
        /// </summary>
        /// <param name="x">The input to process</param>
        /// <param name="z">The output activity on the current layer</param>
        /// <param name="a">The output activation on the current layer</param>
        public abstract void Forward(in FloatSpan2D x, out FloatSpan2D z, out FloatSpan2D a);

        /// <summary>
        /// Backpropagates the error to compute the delta for the inputs of the layer
        /// </summary>
        /// <param name="delta_1">The output error delta</param>
        /// <param name="z">The activity on the inputs of the layer. It will be modified to become the computed delta</param>
        /// <param name="activationPrime">The activation prime function performed by the previous layer</param>
        public abstract void Backpropagate(in FloatSpan2D delta_1, in FloatSpan2D z, ActivationFunction activationPrime);

        #region Equality check

        /// <inheritdoc/>
        public virtual bool Equals(INetworkLayer other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            if (other.GetType() != this.GetType()) return false;
            return other is NetworkLayerBase layer &&
                   Inputs == layer.Inputs &&
                   Outputs == layer.Outputs &&
                   ActivationFunctionType == layer.ActivationFunctionType;
        }

        #endregion

        /// <inheritdoc/>
        public abstract INetworkLayer Clone();
    }
}
