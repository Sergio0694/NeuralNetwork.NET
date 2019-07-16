using System;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.cpuDNN;
using NeuralNetworkDotNet.Helpers;
using NeuralNetworkDotNet.Network.Initialization;
using NeuralNetworkDotNet.Network.Nodes.Unary.Abstract;

namespace NeuralNetworkDotNet.Network.Nodes.Unary
{
    /// <summary>
    /// A batch normalization node, used to improve the convergence speed of a neural network
    /// </summary>
    internal sealed class BatchNormalizationNode : WeightedUnaryNodeBase
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

        public BatchNormalizationNode([NotNull] Node input, NormalizationMode mode) : base(
            input, input.Shape,
            WeightsProvider.NewGammaParameters(input.Shape.C, input.Shape.HW, mode),
            WeightsProvider.NewBetaParameters(input.Shape.C, input.Shape.HW, mode))
        {
            switch (mode)
            {
                case NormalizationMode.Spatial: Mu = Tensor.New(1, input.Shape.C, AllocationMode.Clean); break;
                case NormalizationMode.PerActivation: Mu = Tensor.New(input.Shape, AllocationMode.Clean); break;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Invalid batch normalization mode");
            }

            Sigma2 = Tensor.Like(Mu);
            Sigma2.Span.Fill(1);
            NormalizationMode = mode;
        }

        public BatchNormalizationNode(
            [NotNull] Node input, NormalizationMode mode,
            [NotNull] Tensor w, [NotNull] Tensor b,
            [NotNull] Tensor mu, [NotNull] Tensor sigma2, int iteration)
            : base(input, input.Shape, w, b)
        {
            Mu = mu;
            Sigma2 = sigma2;
            NormalizationMode = mode;
            Iteration = iteration;
        }

        /// <inheritdoc/>
        public override Tensor Forward(Tensor x)
        {
            // TODO: handle inference mode and variable factor
            var y = Tensor.Like(x);
            CpuDnn.BatchNormalizationForward(NormalizationMode, 0.5f, x, Weights, Biases, Mu, Sigma2, y);

            return y;
        }

        /// <inheritdoc/>
        public override Tensor Backward(Tensor x, Tensor y, Tensor dy)
        {
            var dx = Tensor.Like(x);
            CpuDnn.BatchNormalizationBackwardData(NormalizationMode, x, Weights, Mu, Sigma2, dy, dx);

            return dx;
        }

        /// <inheritdoc/>
        public override void Gradient(Tensor x, Tensor dy, out Tensor dJdw, out Tensor dJdb)
        {
            dJdw = Tensor.Like(Weights);
            CpuDnn.BatchNormalizationBackwardGamma(NormalizationMode, x, Mu, Sigma2, dy, dJdw);

            dJdb = Tensor.Like(Biases);
            CpuDnn.BatchNormalizationBackwardBeta(NormalizationMode, dy, dJdb);
        }

        /// <inheritdoc/>
        public override bool Equals(Node other)
        {
            if (!base.Equals(other)) return false;

            return other is BatchNormalizationNode node &&
                   NormalizationMode == node.NormalizationMode &&
                   Iteration == node.Iteration &&
                   Mu.Equals(node.Mu) &&
                   Sigma2.Equals(node.Sigma2);
        }
    }
}
