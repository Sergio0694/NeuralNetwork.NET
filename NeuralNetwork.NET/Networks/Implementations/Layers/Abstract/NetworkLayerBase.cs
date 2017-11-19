using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
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
        [JsonProperty(nameof(Inputs), Required = Required.Always)]
        public abstract int Inputs { get; }

        /// <inheritdoc/>
        [JsonProperty(nameof(Outputs), Required = Required.Always)]
        public abstract int Outputs { get; }

        /// <inheritdoc/>
        [JsonProperty(nameof(ActivationFunctionType), Required = Required.Always)]
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
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public abstract (float[,] Z, float[,] A) Forward([NotNull] float[,] x);

        /// <summary>
        /// Backpropagates the error to compute the delta for the inputs of the layer
        /// </summary>
        /// <param name="delta_1">The output error delta</param>
        /// <param name="z">The activity on the inputs of the layer</param>
        /// <param name="activationPrime">The activation prime function performed by the previous layer</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.ModifyExistingContent)]
        public abstract float[,] Backpropagate([NotNull] float[,] delta_1, [NotNull] float[,] z, ActivationFunction activationPrime);

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
    }
}
