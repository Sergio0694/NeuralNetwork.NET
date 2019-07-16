using System;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Interfaces;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.Helpers;
using NeuralNetworkDotNet.Network.Initialization;
using NeuralNetworkDotNet.Network.Layers.Abstract;

namespace NeuralNetworkDotNet.Network.Layers
{
    /// <summary>
    /// A batch normalization layer, used to improve the convergence speed of a neural network
    /// </summary>
    internal sealed class BatchNormalizationLayer : WeightedLayerBase
    {
        /// <summary>
        /// Gets the mu <see cref="Tensor"/> for the current instance
        /// </summary>
        [NotNull]
        public Tensor Mu { get; }

        /// <summary>
        /// Gets the sigma^2 <see cref="Tensor"/> for the current instance
        /// </summary>
        [NotNull]
        public Tensor Sigma2 { get; }

        /// <summary>
        /// Gets the current iteration number (for the Cumulative Moving Average)
        /// </summary>
        public int Iteration { get; private set; }

        /// <summary>
        /// Gets the current CMA factor used to update the <see cref="Mu"/> and <see cref="Sigma2"/> tensors
        /// </summary>
        public float CumulativeMovingAverageFactor
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => 1f / (1 + Iteration);
        }

        /// <inheritdoc/>
        public override string Hash => Sha256.Hash(Weights.Span).And(Biases.Span).And(Mu.Span).And(Sigma2.Span).ToString();

        /// <summary>
        /// Gets the current normalization mode used in the layer
        /// </summary>
        public NormalizationMode NormalizationMode { get; }

        public BatchNormalizationLayer(Shape shape, NormalizationMode mode) : base(
            shape, shape,
            WeightsProvider.NewGammaParameters(shape.C, shape.HW, mode),
            WeightsProvider.NewBetaParameters(shape.C, shape.HW, mode))
        {
            switch (mode)
            {
                case NormalizationMode.Spatial: Mu = Tensor.New(1, shape.C, AllocationMode.Clean); break;
                case NormalizationMode.PerActivation: Mu = Tensor.New(shape, AllocationMode.Clean); break;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Invalid batch normalization mode");
            }

            Sigma2 = Tensor.Like(Mu);
            Sigma2.Span.Fill(1);
            NormalizationMode = mode;
        }

        public BatchNormalizationLayer(
            Shape shape, NormalizationMode mode,
            [NotNull] Tensor w, [NotNull] Tensor b,
            [NotNull] Tensor mu, [NotNull] Tensor sigma2, int iteration)
            : base(shape, shape, w, b)
        {
            Mu = mu;
            Sigma2 = sigma2;
            NormalizationMode = mode;
            Iteration = iteration;
        }

        /// <inheritdoc/>
        public override Tensor Forward(in Tensor x)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        public override Tensor Backward(Tensor x, Tensor y, Tensor dy)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        public override void Gradient(Tensor x, Tensor dy, out Tensor dJdw, out Tensor dJdb)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        public override bool Equals(ILayer other)
        {
            if (!base.Equals(other)) return false;

            return other is BatchNormalizationLayer layer &&
                   NormalizationMode == layer.NormalizationMode &&
                   Iteration == layer.Iteration &&
                   Mu.Equals(layer.Mu) &&
                   Sigma2.Equals(layer.Sigma2);
        }

        /// <inheritdoc/>
        public override ILayer Clone() => new BatchNormalizationLayer(InputShape, NormalizationMode, Weights.Clone(), Biases.Clone(), Mu.Clone(), Sigma2.Clone(), Iteration);
    }
}
