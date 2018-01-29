using System;
using System.IO;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Layers.Initialization;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Layers.Abstract
{
    /// <summary>
    /// A base claass for a batch normalization layer
    /// </summary>
    internal abstract class BatchNormalizationLayerBase : WeightedLayerBase, IDisposable
    {
        /// <summary>
        /// The cached mu tensor
        /// </summary>
        protected Tensor _Mu;

        /// <summary>
        /// The cached sigma^2 tensor
        /// </summary>
        protected Tensor _Sigma2;

        /// <inheritdoc/>
        public override LayerType LayerType { get; } = LayerType.BatchNormalization;

        /// <summary>
        /// Gets the current normalization mode used in the layer
        /// </summary>
        [JsonProperty(nameof(NormalizationMode), Order = 6)]
        public NormalizationMode NormalizationMode { get; }

        protected BatchNormalizationLayerBase(in TensorInfo shape, NormalizationMode mode, ActivationType activation) 
            : base(shape, shape, 
                WeightsProvider.NewGammaParameters(shape, mode), 
                WeightsProvider.NewBetaParameters(shape, mode), activation)
        {
            NormalizationMode = mode;
        }

        protected BatchNormalizationLayerBase(in TensorInfo shape, NormalizationMode mode, [NotNull] float[] w, [NotNull] float[] b, ActivationType activation) 
            : base(shape, shape, w, b, activation)
        {
            if (w.Length != b.Length) throw new ArgumentException("The size for both gamme and beta paarameters must be the same");
            if (mode == NormalizationMode.Spatial && w.Length != shape.Channels ||
                mode == NormalizationMode.PerActivation && w.Length != shape.Size)
                throw new ArgumentException("Invalid parameters size for the selected normalization mode");
            NormalizationMode = mode;
        }

        /// <summary>
        /// Ensures the temporary <see cref="_Mu"/> and <see cref="_Sigma2"/> tensors are initialized correctly and ready to be used
        /// </summary>
        protected void InitializeNormalizationTensors()
        {
            if (_Mu.IsNull)
            {
                if (!_Sigma2.IsNull) throw new InvalidOperationException();
                switch (NormalizationMode)
                {
                    case NormalizationMode.Spatial:
                        Tensor.New(1, InputInfo.Channels, out _Mu);
                        Tensor.New(1, InputInfo.Channels, out _Sigma2);
                        break;
                    case NormalizationMode.PerActivation:
                        Tensor.New(1, InputInfo.Size, out _Mu);
                        Tensor.New(1, InputInfo.Size, out _Sigma2);
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
        }

        /// <inheritdoc/>
        public override void Serialize(Stream stream)
        {
            base.Serialize(stream);
            stream.Write(NormalizationMode);
        }

        #region IDisposable

        ~BatchNormalizationLayerBase() => Dispose();

        /// <inheritdoc/>
        void IDisposable.Dispose() => Dispose();

        // Disposes the temporary tensors
        private void Dispose()
        {
            _Mu.TryFree();
            _Sigma2.TryFree();
        }

        #endregion
    }
}
