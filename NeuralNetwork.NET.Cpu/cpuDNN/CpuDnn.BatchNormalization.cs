using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.Helpers;

namespace NeuralNetworkDotNet.cpuDNN
{
    public static partial class CpuDnn
    {
        /// <summary>
        /// Gets the minimum epsilon allowed to be used in batch normalization methods
        /// </summary>
        private const float EPSILON = 1e-5f;

        /// <summary>
        /// Executes the forward pass in a batch normalization layer
        /// </summary>
        /// <param name="mode">The desired normalization mode to apply</param>
        /// <param name="factor">The factor for the cumulative moving average</param>
        /// <param name="x">The input <see cref="Tensor"/> to normalize</param>
        /// <param name="gamma">The layer gamma parameters</param>
        /// <param name="beta">The layer beta parameters</param>
        /// <param name="mu">A <see cref="Tensor"/> to use to store the temporary median values (used for backpropagation too)</param>
        /// <param name="sigma2">A <see cref="Tensor"/> to use to store the temporary standard deviation values (used for backpropagation too)</param>
        /// <param name="y">The output <see cref="Tensor"/> for the current layer</param>
        public static void BatchNormalizationForward(
            NormalizationMode mode, float factor,
            [NotNull] Tensor x,
            [NotNull] Tensor gamma, [NotNull] Tensor beta,
            [NotNull] Tensor mu, [NotNull] Tensor sigma2,
            [NotNull] Tensor y)
        {
            Guard.IsTrue(x.Shape == y.Shape, nameof(y), "The output tensor must have the same shape as the input");
            Guard.IsTrue(gamma.Shape == beta.Shape, "The shape of the gamma and beta tensors must be the same");
            Guard.IsTrue(gamma.Shape == mu.Shape, nameof(mu), "The shape of the mu tensor must match gamma");
            Guard.IsTrue(gamma.Shape == sigma2.Shape, nameof(sigma2), "The shape of the sigma2 tensor must match gamma");

            switch (mode)
            {
                // A single mu and variance value per input channel
                case NormalizationMode.Spatial:
                    BatchNormalizationSpatialForward(factor, x, gamma, beta, mu, sigma2, y);
                    break;

                // Each individual activation has its own median and variance
                case NormalizationMode.PerActivation:
                    BatchNormalizationPerActivationForward(factor, x, gamma, beta, mu, sigma2, y);
                    break;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Invalid normalization mode");
            }
        }

        /// <summary>
        /// Executes the forward pass in a batch normalization layer in inference mode
        /// </summary>
        /// <param name="mode">The desired normalization mode to apply</param>
        /// <param name="x">The input <see cref="Tensor"/> to normalize</param>
        /// <param name="gamma">The layer gamma parameters</param>
        /// <param name="beta">The layer beta parameters</param>
        /// <param name="mu">A <see cref="Tensor"/> to use to store the temporary median values (used for backpropagation too)</param>
        /// <param name="sigma2">A <see cref="Tensor"/> to use to store the temporary standard deviation values (used for backpropagation too)</param>
        /// <param name="y">The output <see cref="Tensor"/> for the current layer</param>
        public static void BatchNormalizationForward(
            NormalizationMode mode,
            [NotNull] Tensor x,
            [NotNull] Tensor gamma, [NotNull] Tensor beta,
            [NotNull] Tensor mu, [NotNull] Tensor sigma2,
            [NotNull] Tensor y)
        {
            Guard.IsTrue(x.Shape == y.Shape, nameof(y), "The output tensor must have the same shape as the input");
            Guard.IsTrue(gamma.Shape == beta.Shape, "The shape of the gamma and beta tensors must be the same");
            Guard.IsTrue(gamma.Shape == mu.Shape, nameof(mu), "The shape of the mu tensor must match gamma");
            Guard.IsTrue(gamma.Shape == sigma2.Shape, nameof(sigma2), "The shape of the sigma2 tensor must match gamma");

            switch (mode)
            {
                case NormalizationMode.Spatial: BatchNormalizationSpatialForward(x, mu, sigma2, gamma, beta, y); break;
                case NormalizationMode.PerActivation: BatchNormalizationPerActivationForward(x, mu, sigma2, gamma, beta, y); break;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Invalid normalization mode");
            }
        }

        /// <summary>
        /// Executes the backward pass through a batch normalization layer
        /// </summary>
        /// <param name="mode">The desired normalization mode to apply</param>
        /// <param name="x">The input <see cref="Tensor"/> to normalize</param>
        /// <param name="gamma">The layer gamma parameters</param>
        /// <param name="mu">A <see cref="Tensor"/> with the temporary median values calculated in the forward pass</param>
        /// <param name="sigma2">A <see cref="Tensor"/> with the temporary standard deviation values calculated in the forward pass</param>
        /// <param name="dy">The output error delta <see cref="Tensor"/></param>
        /// <param name="dx">The resulting backpropagated error delta <see cref="Tensor"/></param>
        public static void BatchNormalizationBackwardData(
            NormalizationMode mode,
            [NotNull] Tensor x,
            [NotNull] Tensor gamma,
            [NotNull] Tensor mu, [NotNull] Tensor sigma2,
            [NotNull] Tensor dy, [NotNull] Tensor dx)
        {

            Guard.IsTrue(x.Shape == dy.Shape, "The input and output tensors must have the same shape");
            Guard.IsTrue(x.Shape == dx.Shape, "The input the resulting error tensor must have the same shape");
            Guard.IsTrue(gamma.Shape == mu.Shape, nameof(mu), "The shape of the mu tensor must match gamma");
            Guard.IsTrue(gamma.Shape == sigma2.Shape, nameof(sigma2), "The shape of the sigma2 tensor must match gamma");

            switch (mode)
            {
                case NormalizationMode.Spatial: BatchNormalizationSpatialBackwardData(x, mu, sigma2, gamma, dy, dx); break;
                case NormalizationMode.PerActivation: BatchNormalizationPerActivationBackwardData(x, mu, sigma2, gamma, dy, dx); break;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Invalid normalization mode");
            }
        }

        /// <summary>
        /// Calculates the gradient with respect to the gamma <see cref="Tensor"/> in a batch normalization layer
        /// </summary>
        /// <param name="mode">The desired normalization mode to apply</param>
        /// <param name="x">The input <see cref="Tensor"/> used in the forward pass</param>
        /// <param name="mu">A <see cref="Tensor"/> with the temporary median values calculated in the forward pass</param>
        /// <param name="sigma2">A <see cref="Tensor"/> with the temporary standard deviation values calculated in the forward pass</param>
        /// <param name="dy">The output <see cref="Tensor"/> error delta for the current layer</param>
        /// <param name="dgamma">The resulting gamma gradient <see cref="Tensor"/></param>
        public static void BatchNormalizationBackwardGamma(
            NormalizationMode mode,
            [NotNull] Tensor x,
            [NotNull] Tensor mu, [NotNull] Tensor sigma2,
            [NotNull] Tensor dy, [NotNull] Tensor dgamma)
        {
            Guard.IsTrue(x.Shape == dy.Shape, "The input and output tensors must have the same shape");
            Guard.IsTrue(dgamma.Shape == mu.Shape, nameof(mu), "The shape of the mu tensor must match gamma");
            Guard.IsTrue(dgamma.Shape == sigma2.Shape, nameof(sigma2), "The shape of the sigma2 tensor must match gamma");

            switch (mode)
            {
                case NormalizationMode.Spatial: BatchNormalizationSpatialBackwardGamma(x, mu, sigma2, dy, dgamma); break;
                case NormalizationMode.PerActivation: BatchNormalizationPerActivationBackwardGamma(x, mu, sigma2, dy, dgamma); break;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Invalid normalization mode");
            }
        }

        /// <summary>
        /// Calculates the gradient with respect to the beta <see cref="Tensor"/> in a batch normalization layer
        /// </summary>
        /// <param name="mode">The desired normalization mode to apply</param>
        /// <param name="dy">The output <see cref="Tensor"/> error delta for the current layer</param>
        /// <param name="dbeta">The resulting beta gradient <see cref="Tensor"/></param>
        public static void BatchNormalizationBackwardBeta(NormalizationMode mode, [NotNull] Tensor dy, [NotNull] Tensor dbeta)
        {
            switch (mode)
            {
                case NormalizationMode.Spatial: BatchNormalizationSpatialBackwardBeta(dy, dbeta); break;
                case NormalizationMode.PerActivation: FullyConnectedBackwardBias(dy, dbeta); break;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Invalid normalization mode");
            }
        }

        #region Spatial

        // Spatial forward training batch normalization
        private static void BatchNormalizationSpatialForward(
            float factor,
            [NotNull] Tensor x,
            [NotNull] Tensor gamma, [NotNull] Tensor beta,
            [NotNull]Tensor mu, [NotNull] Tensor sigma2,
            [NotNull] Tensor y)
        {
            Guard.IsTrue(mu.Shape == (1, x.Shape.C), "Invalid mu tensor size");

            int
                n = x.Shape.N,
                l = x.Shape.CHW,
                nhw = x.Shape.N * x.Shape.HW,
                slice = x.Shape.HW;

            // Mean and variance
            Parallel.For(0, x.Shape.C, c =>
            {
                // Mu
                var start = slice * c;
                var mc = 0f;

                ref var rx = ref x.Span.GetPinnableReference();

                for (var i = 0; i < n; i++)
                {
                    var offset = start + i * l;
                    for (var xy = 0; xy < slice; xy++)
                        mc += Unsafe.Add(ref rx, offset + xy);
                }

                mu.Span[c] = mc /= nhw * factor + mu.Span[c] * (1 - factor);

                // Variance
                var sc = 0f;
                for (var i = 0; i < n; i++)
                {
                    var offset = start + i * l;
                    for (var xy = 0; xy < slice; xy++)
                    {
                        var sq = Unsafe.Add(ref rx, offset + xy) - mc;
                        sc += sq * sq;
                    }
                }

                sigma2.Span[c] = sc / nhw * factor + sigma2.Span[c] * (1 - factor);

            });

            // Normalization
            Parallel.For(0, x.Shape.C, c =>
            {
                float
                    gc = gamma.Span[c],
                    bc = beta.Span[c],
                    mc = mu.Span[c],
                    sqrt_1 = 1 / (float)Math.Sqrt(sigma2.Span[c] + EPSILON);
                var start = slice * c;

                ref var rx = ref x.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                for (var i = 0; i < n; i++)
                {
                    var offset = start + i * l;
                    for (var xy = 0; xy < slice; xy++)
                    {
                        var hat = (Unsafe.Add(ref rx, offset + xy) - mc) * sqrt_1;
                        Unsafe.Add(ref ry, offset + xy) = gc * hat + bc;
                    }
                }
            });
        }

        // Spatial forward inference batch normalization
        private static void BatchNormalizationSpatialForward(
            [NotNull] Tensor x,
            [NotNull] Tensor gamma, [NotNull] Tensor beta,
            [NotNull] Tensor mu, [NotNull] Tensor sigma2,
            [NotNull] Tensor y)
        {
            Guard.IsTrue(mu.Shape == (1, x.Shape.C), "Invalid mu tensor size");

            int
                n = x.Shape.N,
                l = x.Shape.CHW,
                slice = x.Shape.HW;

            Parallel.For(0, x.Shape.C, c =>
            {
                float
                    gc = gamma.Span[c],
                    bc = beta.Span[c],
                    mc = mu.Span[c],
                    sqrt_1 = 1 / (float)Math.Sqrt(sigma2.Span[c] + EPSILON);
                var start = slice * c;

                ref var rx = ref x.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                for (var i = 0; i < n; i++)
                {
                    var offset = start + i * l;
                    for (var xy = 0; xy < slice; xy++)
                    {
                        var hat = (Unsafe.Add(ref rx, offset + xy) - mc) * sqrt_1;
                        Unsafe.Add(ref ry, offset + xy) = gc * hat + bc;
                    }
                }
            });
        }

        // Spatial backward batch normalization
        private static void BatchNormalizationSpatialBackwardData(
            [NotNull] Tensor x,
            [NotNull] Tensor gamma,
            [NotNull] Tensor mu, [NotNull] Tensor sigma2,
            [NotNull] Tensor dy, Tensor dx)
        {
            Guard.IsTrue(mu.Shape == (1, x.Shape.C), "Invalid mu tensor size");

            int
                n = dx.Shape.N,
                l = dx.Shape.CHW,
                nhw = x.Shape.N * x.Shape.HW,
                slice = x.Shape.HW;

            Parallel.For(0, x.Shape.C, c =>
            {
                // Calculate the two summatories
                float
                    mc = mu.Span[c],
                    sc = sigma2.Span[c],
                    left = 1f / nhw * gamma.Span[c] / (float)Math.Sqrt(sigma2.Span[c] + EPSILON),
                    _2nd = 0,
                    _3rdRight = 0;
                var start = slice * c;

                ref var rx = ref x.Span.GetPinnableReference();
                ref var rdx = ref dx.Span.GetPinnableReference();
                ref var rdy = ref dy.Span.GetPinnableReference();


                for (var i = 0; i < n; i++, start += l)
                for (var xy = 0; xy < slice; xy++)
                {
                    var pdyicxy = Unsafe.Add(ref rdy, start + xy);
                    _2nd += pdyicxy;
                    _3rdRight += pdyicxy * (Unsafe.Add(ref rx, start + xy) - mc);
                }

                // Assign the backpropagated tensor
                start = slice * c;
                for (var i = 0; i < n; i++, start += l)
                for (var xy = 0; xy < slice; xy++)
                {
                        Unsafe.Add(ref rdx, start + xy) =
                            left * (nhw * Unsafe.Add(ref rdy, start + xy) -
                                    _2nd - (Unsafe.Add(ref rx, start + xy) - mc) / (sc + EPSILON) * _3rdRight);
                }
            });
        }

        // Spatial batch normalization gamma gradient
        private static void BatchNormalizationSpatialBackwardGamma(
            [NotNull] Tensor x,
            [NotNull] Tensor mu, [NotNull] Tensor sigma2,
            [NotNull] Tensor dy, [NotNull] Tensor dgamma)
        {
            Guard.IsTrue(mu.Shape == (1, x.Shape.C), "Invalid mu tensor size");

            int
                n = x.Shape.N,
                l = x.Shape.CHW,
                slice = x.Shape.HW;

            Parallel.For(0, x.Shape.C, c =>
            {
                float gc = 0, mc = mu.Span[c], sc = (float)Math.Sqrt(sigma2.Span[c] + EPSILON);
                var offset = slice * c;

                ref var rx = ref x.Span.GetPinnableReference();
                ref var rdy = ref dy.Span.GetPinnableReference();

                for (var i = 0; i < n; i++, offset += l)
                    for (var xy = 0; xy < slice; xy++)
                        gc += Unsafe.Add(ref rdy, offset + xy) * (Unsafe.Add(ref rx, offset + xy) - mc) / sc;

                dgamma.Span[c] = gc;
            });
        }

        // Spatial batch normalization beta gradient
        private static void BatchNormalizationSpatialBackwardBeta([NotNull] Tensor dy, [NotNull] Tensor dbeta)
        {
            Guard.IsTrue(dbeta.Shape == (1, dy.Shape.C), "The beta tensor must have a value for each input channel");

            int
                n = dy.Shape.N,
                slice = dy.Shape.HW,
                l = dy.Shape.CHW;

            // Accumulate the output gradient
            Parallel.For(0, dy.Shape.C, c =>
            {
                var bc = 0f;
                var start = c * slice;

                ref var rdy = ref dy.Span.GetPinnableReference();

                for (var i = 0; i < n; i++, start += l)
                    for (var xy = 0; xy < slice; xy++)
                        bc += Unsafe.Add(ref rdy, start + xy);

                dbeta.Span[c] = bc;
            });
        }

        #endregion

        #region Per activation

        // Per-activation forward training batch normalization
        private static void BatchNormalizationPerActivationForward(
            float factor,
            [NotNull] Tensor x,
            [NotNull] Tensor gamma, [NotNull] Tensor beta,
            [NotNull] Tensor mu, [NotNull] Tensor sigma2,
            [NotNull] Tensor y)
        {
            Guard.IsTrue(mu.Shape == (1, x.Shape.C), "Invalid mu tensor size");

            int
                n = x.Shape.N,
                l = x.Shape.CHW;

            Parallel.For(0, l, j =>
            {
                var mi = 0f;
                ref var rx = ref x.Span.GetPinnableReference();

                // Mean
                for (var i = 0; i < n; i++)
                    mi += Unsafe.Add(ref rx, i * l + j);

                mu.Span[j] = mi /= n * factor + mu.Span[j] * (1 - factor);

                // Variance
                var sl = 0f;
                for (var i = 0; i < n; i++)
                {
                    var hm = Unsafe.Add(ref rx, i * l + j) - mi;
                    sl += hm * hm;
                }

                sigma2.Span[j] = sl / n * factor + sigma2.Span[j] * (1 - factor);

            });

            // Apply the batch normalization pass
            Parallel.For(0, n, i =>
            {
                var offset = i * l;

                ref var rx = ref x.Span.GetPinnableReference();
                ref var rgamma = ref gamma.Span.GetPinnableReference();
                ref var rbeta = ref beta.Span.GetPinnableReference();
                ref var rmu = ref mu.Span.GetPinnableReference();
                ref var rsigma2 = ref sigma2.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                for (var j = 0; j < l; j++)
                {
                    var hat =
                        (Unsafe.Add(ref rx, offset + j) - Unsafe.Add(ref rmu, j)) /
                        (float)Math.Sqrt(Unsafe.Add(ref rsigma2, j) + EPSILON);
                    Unsafe.Add(ref ry, offset + j) = Unsafe.Add(ref rgamma, j) * hat + Unsafe.Add(ref rbeta, j);
                }
            });
        }

        // Per-activation forward inference batch normalization
        private static void BatchNormalizationPerActivationForward(
            [NotNull] Tensor x,
            [NotNull] Tensor gamma, [NotNull] Tensor beta,
            [NotNull] Tensor mu, [NotNull] Tensor sigma2,
            [NotNull] Tensor y)
        {
            Guard.IsTrue(mu.Shape == (1, x.Shape.C), "Invalid mu tensor size");

            int
                n = x.Shape.N,
                l = x.Shape.CHW;

            Parallel.For(0, n, i =>
            {
                var offset = i * l;

                ref var rx = ref x.Span.GetPinnableReference();
                ref var rgamma = ref gamma.Span.GetPinnableReference();
                ref var rbeta = ref beta.Span.GetPinnableReference();
                ref var rmu = ref mu.Span.GetPinnableReference();
                ref var rsigma2 = ref sigma2.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                for (var j = 0; j < l; j++)
                {
                    var hat =
                        (Unsafe.Add(ref rx, offset + j) - Unsafe.Add(ref rmu, j)) /
                        (float)Math.Sqrt(Unsafe.Add(ref rsigma2, j) + EPSILON);
                    Unsafe.Add(ref ry, offset + j) = Unsafe.Add(ref rgamma, j) * hat + Unsafe.Add(ref rbeta, j);
                }
            });
        }

        // Per-activation backward batch normalization
        private static void BatchNormalizationPerActivationBackwardData(
            [NotNull] Tensor x,
            [NotNull] Tensor gamma,
            [NotNull] Tensor mu, [NotNull] Tensor sigma2,
            [NotNull] Tensor dy, [NotNull] Tensor dx)
        {
            Guard.IsTrue(mu.Shape == (1, x.Shape.C), "Invalid mu tensor size");

            int
                n = dx.Shape.N,
                l = dx.Shape.CHW;

            Parallel.For(0, n, i =>
            {
                ref var rx = ref x.Span.GetPinnableReference();
                ref var rgamma = ref gamma.Span.GetPinnableReference();
                ref var rmu = ref mu.Span.GetPinnableReference();
                ref var rsigma2 = ref sigma2.Span.GetPinnableReference();
                ref var rdy = ref dy.Span.GetPinnableReference();
                ref var rdx = ref dx.Span.GetPinnableReference();

                for (var j = 0; j < l; j++)
                {
                    float
                        jmu = Unsafe.Add(ref rmu, j),
                        left = 1f / n * Unsafe.Add(ref rgamma, j) / (float)Math.Sqrt(Unsafe.Add(ref rsigma2, j) + EPSILON),
                        _1st = n * Unsafe.Add(ref rdy, i * l + j),
                        _2nd = 0,
                        _3rdLeft = (Unsafe.Add(ref rx, i * l + j) - jmu) / (Unsafe.Add(ref rsigma2, j) + EPSILON),
                        _3rdRight = 0;

                    for (var k = 0; k < n; k++)
                    {
                        var pdykj = Unsafe.Add(ref rdy, k * l + j);
                        _2nd += pdykj;
                        _3rdRight += pdykj * (Unsafe.Add(ref rx, k * l + j) - jmu);
                    }

                    Unsafe.Add(ref rdx, i * l + j) = left * (_1st - _2nd - _3rdLeft * _3rdRight);
                }
            });
        }

        // Per-activation batch normalization gamma gradient
        private static void BatchNormalizationPerActivationBackwardGamma(
            [NotNull] Tensor x,
            [NotNull] Tensor mu, [NotNull] Tensor sigma2,
            [NotNull] Tensor dy, [NotNull] Tensor dgamma)
        {
            Guard.IsTrue(mu.Shape == (1, x.Shape.C), "Invalid mu tensor size");

            int
                n = x.Shape.N,
                l = x.Shape.CHW;

            Parallel.For(0, x.Shape.CHW, j =>
            {
                float sum = 0, sj = (float)Math.Sqrt(sigma2.Span[j] + EPSILON);

                ref var rx = ref x.Span.GetPinnableReference();
                ref var rmu = ref mu.Span.GetPinnableReference();
                ref var rdy = ref dy.Span.GetPinnableReference();

                for (var i = 0; i < n; i++)
                {
                    var hat = (Unsafe.Add(ref rx, i * l + j) - Unsafe.Add(ref rmu, j)) / sj;
                    sum += Unsafe.Add(ref rdy, i * l + j) * hat;
                }

                dgamma.Span[j] = sum;
            });
        }

        #endregion
    }
}
