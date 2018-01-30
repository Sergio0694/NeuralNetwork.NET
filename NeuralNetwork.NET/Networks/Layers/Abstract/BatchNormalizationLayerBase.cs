using System;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Settings;
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
        #region Fields and parameters

        /// <summary>
        /// The cached mu tensor
        /// </summary>
        protected readonly Tensor Mu;

        /// <summary>
        /// The cached sigma^2 tensor
        /// </summary>
        protected readonly Tensor Sigma2;

        // The current iteration number (for the Cumulative Moving Average)
        private int _Iteration;

        /// <inheritdoc/>
        public override LayerType LayerType { get; } = LayerType.BatchNormalization;

        /// <summary>
        /// Gets the current normalization mode used in the layer
        /// </summary>
        [JsonProperty(nameof(NormalizationMode), Order = 6)]
        public NormalizationMode NormalizationMode { get; }

        #endregion

        protected BatchNormalizationLayerBase(in TensorInfo shape, NormalizationMode mode, ActivationType activation) 
            : base(shape, shape, 
                WeightsProvider.NewGammaParameters(shape, mode), 
                WeightsProvider.NewBetaParameters(shape, mode), activation)
        {
            switch (mode)
            {
                case NormalizationMode.Spatial:
                    Tensor.New(1, InputInfo.Channels, out Mu);
                    Tensor.New(1, InputInfo.Channels, out Sigma2);
                    break;
                case NormalizationMode.PerActivation:
                    Tensor.New(1, InputInfo.Size, out Mu);
                    Tensor.New(1, InputInfo.Size, out Sigma2);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
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

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            if (NetworkSettings.TrainingInProgress) ForwardTraining(1f / (1 + _Iteration++), x, out z, out a);
            else ForwardInference(x, out z, out a);
        }

        /// <summary>
        /// Forwards the inputs through the batch normalization layer during an inference pass
        /// </summary>
        /// <param name="x">The input to process</param>
        /// <param name="z">The output activity on the current layer</param>
        /// <param name="a">The output activation on the current layer</param>
        public abstract void ForwardInference(in Tensor x, out Tensor z, out Tensor a);

        /// <summary>
        /// Forwards the inputs through the batch normalization layer during a training pass, updating the CMA mean and variance <see cref="Tensor"/> instances
        /// </summary>
        /// <param name="factor">The factor to use to update the cumulative moving average</param>
        /// <param name="x">The input to process</param>
        /// <param name="z">The output activity on the current layer</param>
        /// <param name="a">The output activation on the current layer</param>
        public abstract void ForwardTraining(float factor, in Tensor x, out Tensor z, out Tensor a);

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
        [SuppressMessage("ReSharper", "ImpureMethodCallOnReadonlyValueField")]
        private void Dispose()
        {
            Mu.Free();
            Sigma2.Free();
        }

        #endregion
    }
}
