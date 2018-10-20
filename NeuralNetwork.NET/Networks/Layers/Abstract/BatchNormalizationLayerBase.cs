using System;
using System.IO;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Layers.Initialization;
using NeuralNetworkNET.SupervisedLearning.Optimization;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Layers.Abstract
{
    /// <summary>
    /// A base claass for a batch normalization layer
    /// </summary>
    internal abstract class BatchNormalizationLayerBase : WeightedLayerBase
    {
        #region Fields and parameters

        /// <summary>
        /// The cached mu tensor
        /// </summary>
        [NotNull]
        public float[] Mu { get; }

        /// <summary>
        /// The cached sigma^2 tensor
        /// </summary>
        [NotNull]
        public float[] Sigma2 { get; }

        /// <summary>
        /// Gets the current iteration number (for the Cumulative Moving Average)
        /// </summary>
        public int Iteration { get; private set; }

        /// <summary>
        /// Gets the current CMA factor used to update the <see cref="Mu"/> and <see cref="Sigma2"/> tensors
        /// </summary>
        [JsonProperty(nameof(CumulativeMovingAverageFactor), Order = 6)]
        public float CumulativeMovingAverageFactor
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => 1f / (1 + Iteration);
        }

        /// <inheritdoc/>
        public override string Hash => Convert.ToBase64String(Sha256.Hash(Weights, Biases, Mu, Sigma2));

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
                    Mu = new float[InputInfo.Channels];
                    Sigma2 = new float[InputInfo.Channels];
                    break;
                case NormalizationMode.PerActivation:
                    Mu = new float[InputInfo.Size];
                    Sigma2 = new float[InputInfo.Size];
                    break;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Invalid batch normalization mode");
            }
            Sigma2.AsSpan().Fill(1);
            NormalizationMode = mode;
        }

        protected BatchNormalizationLayerBase(in TensorInfo shape, NormalizationMode mode, [NotNull] float[] w, [NotNull] float[] b, int iteration, [NotNull] float[] mu, [NotNull] float[] sigma2, ActivationType activation) 
            : base(shape, shape, w, b, activation)
        {
            if (w.Length != b.Length) throw new ArgumentException("The size for both gamme and beta paarameters must be the same");
            if (mode == NormalizationMode.Spatial && w.Length != shape.Channels ||
                mode == NormalizationMode.PerActivation && w.Length != shape.Size)
                throw new ArgumentException("Invalid parameters size for the selected normalization mode");
            if (iteration < 0) throw new ArgumentOutOfRangeException(nameof(iteration), "The iteration value must be aat least equal to 0");
            if (mu.Length != w.Length || sigma2.Length != w.Length)
                throw new ArgumentException("The mu and sigma2 parameters must match the shape of the gamma and beta parameters");
            NormalizationMode = mode;
            Iteration = iteration;
            Mu = mu;
            Sigma2 = sigma2;
        }

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            if (NetworkTrainer.BackpropagationInProgress) ForwardTraining(1f / (1 + Iteration++), x, out z, out a);
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
        public override bool Equals(INetworkLayer other)
        {
            if (!base.Equals(other)) return false;
            return other is BatchNormalizationLayerBase layer &&
                   Iteration == layer.Iteration &&
                   Mu.ContentEquals(layer.Mu) &&
                   Sigma2.ContentEquals(layer.Sigma2);
        }

        /// <inheritdoc/>
        public override void Serialize(Stream stream)
        {
            base.Serialize(stream);
            stream.Write(NormalizationMode);
            stream.Write(Iteration);
            stream.Write(Mu.Length);
            stream.WriteShuffled(Mu);
            stream.Write(Sigma2.Length);
            stream.WriteShuffled(Sigma2);
        }
    }
}
