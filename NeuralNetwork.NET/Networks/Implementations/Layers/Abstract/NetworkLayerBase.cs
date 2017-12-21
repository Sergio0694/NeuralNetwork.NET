using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Misc;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using Newtonsoft.Json;
using System.IO;

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
        [JsonProperty(nameof(LayerType), Order = 1)]
        public abstract LayerType LayerType { get; }

        /// <inheritdoc/>
        [JsonProperty(nameof(InputInfo), Order = 2)]
        public TensorInfo InputInfo { get; }

        /// <inheritdoc/>
        [JsonProperty(nameof(OutputInfo), Order = 3)]
        public TensorInfo OutputInfo { get; }

        /// <summary>
        /// Gets the activation type used in the current layer
        /// </summary>
        [JsonProperty(nameof(ActivationFunctionType), Order = 4)]
        public ActivationFunctionType ActivationFunctionType { get; }

        /// <summary>
        /// Gets the list of activation and activation prime functions used in the network
        /// </summary>
        public (ActivationFunction Activation, ActivationFunction ActivationPrime) ActivationFunctions { get; }

        #endregion

        protected NetworkLayerBase(in TensorInfo input, in TensorInfo output, ActivationFunctionType activation)
        {
            InputInfo = input;
            OutputInfo = output;
            ActivationFunctionType = activation;
            ActivationFunctions = ActivationFunctionProvider.GetActivations(activation);
        }

        /// <summary>
        /// Forwards the inputs through the network layer and returns the resulting activity (Z) and activation (A)
        /// </summary>
        /// <param name="x">The input to process</param>
        /// <param name="z">The output activity on the current layer</param>
        /// <param name="a">The output activation on the current layer</param>
        public abstract void Forward(in Tensor x, out Tensor z, out Tensor a);

        /// <summary>
        /// Backpropagates the error to compute the delta for the inputs of the layer
        /// </summary>
        /// <param name="delta_1">The output error delta</param>
        /// <param name="z">The activity on the inputs of the layer. It will be modified to become the computed delta</param>
        /// <param name="activationPrime">The activation prime function performed by the previous layer</param>
        public abstract void Backpropagate(in Tensor delta_1, in Tensor z, ActivationFunction activationPrime);

        #region Equality check

        /// <inheritdoc/>
        public virtual bool Equals(INetworkLayer other)
        {
            if (other is null) return false;
            if (ReferenceEquals(this, other)) return true;
            if (other.GetType() != GetType()) return false;
            return other is NetworkLayerBase layer &&
                   InputInfo == layer.InputInfo &&
                   OutputInfo == layer.OutputInfo &&
                   ActivationFunctionType == layer.ActivationFunctionType;
        }

        #endregion

        /// <inheritdoc/>
        public abstract INetworkLayer Clone();

        /// <summary>
        /// Writes the current layer to the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The target <see cref="Stream"/> to use to write the layer data</param>
        public virtual void Serialize([NotNull] Stream stream)
        {
            stream.Write(LayerType);
            stream.Write(InputInfo);
            stream.Write(OutputInfo);
            stream.Write(ActivationFunctionType);
        }
    }
}
