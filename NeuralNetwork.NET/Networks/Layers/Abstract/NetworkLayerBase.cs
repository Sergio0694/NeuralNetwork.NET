using System;
using System.IO;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Layers.Abstract
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

        [JsonProperty(nameof(InputInfo), Order = 2)]
        private readonly TensorInfo _InputInfo;

        /// <inheritdoc/>
        public ref readonly TensorInfo InputInfo
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _InputInfo;
        }

        [JsonProperty(nameof(OutputInfo), Order = 3)]
        public readonly TensorInfo _OutputInfo;

        /// <inheritdoc/>
        public ref readonly TensorInfo OutputInfo
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _OutputInfo;
        }

        /// <summary>
        /// Gets the activation type used in the current layer
        /// </summary>
        [JsonProperty(nameof(ActivationType), Order = 4)]
        public ActivationType ActivationType { get; }

        /// <summary>
        /// Gets the list of activation and activation prime functions used in the network
        /// </summary>
        protected readonly (ActivationFunction Activation, ActivationFunction ActivationPrime) ActivationFunctions;

        #endregion

        protected NetworkLayerBase(in TensorInfo input, in TensorInfo output, ActivationType activation)
        {
            _InputInfo = input.IsEmptyOrInvalid ? throw new ArgumentException("The layer input info is not valid", nameof(input)) : input;
            _OutputInfo = output.IsEmptyOrInvalid ? throw new ArgumentException("The layer output info is not valid", nameof(output)) : output;
            ActivationType = activation;
            ActivationFunctions = ActivationFunctionProvider.GetActivations(activation);
        }

        /// <summary>
        /// Forwards the inputs through the network layer and returns the resulting activity (Z) and activation (A)
        /// </summary>
        /// <param name="x">The input to process</param>
        /// <param name="z">The output activity on the current layer</param>
        /// <param name="a">The output activation on the current layer</param>
        public abstract void Forward(in Tensor x, out Tensor z, out Tensor a);

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
                   ActivationType == layer.ActivationType;
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
            stream.Write(ActivationType);
        }
    }
}
