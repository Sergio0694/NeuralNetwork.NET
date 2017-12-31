using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A simple map that stores references to <see cref="Tensor"/> instances while training or using a network
    /// </summary>
    internal sealed class TensorMap : IDisposable
    {
        #region IDisposable

        ~TensorMap() => Dispose();

        /// <inheritdoc/>
        void IDisposable.Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose();
        }

        // Frees the allocated tensors
        private void Dispose()
        {
            foreach (Tensor tensor in new[] { ActivityMap, ActivationMap, DeltaMap }.SelectMany(d => d.Values))
                tensor.Free();
            ActivityMap.Clear();
            ActivationMap.Clear();
            DeltaMap.Clear();
        }

        #endregion

        // The Z tensors
        [NotNull]
        private readonly IDictionary<INetworkLayer, Tensor> ActivityMap = new Dictionary<INetworkLayer, Tensor>();

        // The A tensors
        [NotNull]
        private readonly IDictionary<INetworkLayer, Tensor> ActivationMap = new Dictionary<INetworkLayer, Tensor>();

        // The dy tensors
        [NotNull]
        private readonly IDictionary<INetworkLayer, Tensor> DeltaMap = new Dictionary<INetworkLayer, Tensor>();

        /// <summary>
        /// Gets or sets a <see cref="Tensor"/> for the given network layer and data type
        /// </summary>
        /// <param name="layer">The source <see cref="INetworkLayer"/> instance for the target <see cref="Tensor"/></param>
        /// <param name="type">The <see cref="TensorType"/> value for the target <see cref="Tensor"/></param>
        public Tensor this[INetworkLayer layer, TensorType type]
        {
            [Pure]
            get
            {
                switch (type)
                {
                    case TensorType.Activity: return ActivityMap[layer];
                    case TensorType.Activation: return ActivationMap[layer];
                    case TensorType.Delta: return DeltaMap[layer];
                    default: throw new ArgumentOutOfRangeException(nameof(type), "Invalid data type requested");
                }
            }
            set
            {
                IDictionary<INetworkLayer, Tensor> target;
                switch (type)
                {
                    case TensorType.Activity: target = ActivityMap; break;
                    case TensorType.Activation: target = ActivationMap; break;
                    case TensorType.Delta: target = DeltaMap; break;
                    default: throw new ArgumentOutOfRangeException(nameof(type), "Invalid data type requested");
                }
                if (target.TryGetValue(layer, out Tensor old)) old.Free();
                target[layer] = value;
            }
        }
    }

    /// <summary>
    /// Indicates the type of any given <see cref="Tensor"/>
    /// </summary>
    internal enum TensorType
    {
        /// <summary>
        /// The activity of a network layer, the output before the activation function
        /// </summary>
        Activity,

        /// <summary>
        /// The activation of a network layer, the output with the activation function applied to it
        /// </summary>
        Activation,

        /// <summary>
        /// The error delta for the outputs of a given network layer
        /// </summary>
        Delta
    }
}
