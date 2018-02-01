using System;
using System.Threading.Tasks;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.cpuDNN
{
    public static partial class CpuDnn
    {
        /// <summary>
        /// Gets the minimum epsilon allowed to be used in batch normalization methods
        /// </summary>
        internal static readonly float CUDNN_BN_MIN_EPSILON = 1e-5.ToApproximatedFloat();

        /// <summary>
        /// Executes the forward pass in a batch normalization layer
        /// </summary>
        /// <param name="mode">The desired normalization mode to apply</param>
        /// <param name="info">The ifo on the input <see cref="Tensor"/> to process</param>
        /// <param name="x">The input <see cref="Tensor"/> to normalize</param>
        /// <param name="factor">The factor for the cumulative moving average</param>
        /// <param name="mu">A <see cref="Tensor"/> to use to store the temporary median values (used for backpropagation too)</param>
        /// <param name="sigma2">A <see cref="Tensor"/> to use to store the temporary standard deviation values (used for backpropagation too)</param>
        /// <param name="gamma">The layer gamma parameters</param>
        /// <param name="beta">The layer beta parameters</param>
        /// <param name="y">The output <see cref="Tensor"/> for the current layer</param>
        public static void BatchNormalizationForward(
            NormalizationMode mode, in TensorInfo info, in Tensor x, 
            float factor, in Tensor mu, in Tensor sigma2, 
            in Tensor gamma, in Tensor beta, in Tensor y)
        {
            if (info.Size != x.Length) throw new ArgumentException("The tensor info doesn't match the length of the input tensor", nameof(x));
            if (!sigma2.MatchShape(mu)) throw new ArgumentException("Invalid standard deviation tensor shape", nameof(sigma2));
            if (!gamma.MatchShape(sigma2)) throw new ArgumentException("The gamma tensor doesn't have the right shape", nameof(gamma));
            if (!beta.MatchShape(gamma)) throw new ArgumentException("The beta tensor doesn't have the right shape", nameof(beta));
            if (!x.MatchShape(y)) throw new ArgumentException("The input and output tensors must have the same shape", nameof(y));
            switch (mode)
            {
                // A single mu and variance value per input channel
                case NormalizationMode.Spatial:
                    BatchNormalizationForward(info, x, factor, mu, sigma2, gamma, beta, y);
                    break;

                // Each individual activation has its own median and variance
                case NormalizationMode.PerActivation:
                    BatchNormalizationForward(x, factor, mu, sigma2, gamma, beta, y);
                    break;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Invalid normalization mode");
            }
        }

        /// <summary>
        /// Executes the forward pass in a batch normalization layer in inference mode
        /// </summary>
        /// <param name="mode">The desired normalization mode to apply</param>
        /// <param name="info">The ifo on the input <see cref="Tensor"/> to process</param>
        /// <param name="x">The input <see cref="Tensor"/> to normalize</param>
        /// <param name="mu">A <see cref="Tensor"/> to use to store the temporary median values (used for backpropagation too)</param>
        /// <param name="sigma2">A <see cref="Tensor"/> to use to store the temporary standard deviation values (used for backpropagation too)</param>
        /// <param name="gamma">The layer gamma parameters</param>
        /// <param name="beta">The layer beta parameters</param>
        /// <param name="y">The output <see cref="Tensor"/> for the current layer</param>
        public static void BatchNormalizationForward(
            NormalizationMode mode, in TensorInfo info, in Tensor x, 
            in Tensor mu, in Tensor sigma2, 
            in Tensor gamma, in Tensor beta, in Tensor y)
        {
            if (info.Size != x.Length) throw new ArgumentException("The tensor info doesn't match the length of the input tensor", nameof(x));
            if (!sigma2.MatchShape(mu)) throw new ArgumentException("Invalid standard deviation tensor shape", nameof(sigma2));
            if (!gamma.MatchShape(sigma2)) throw new ArgumentException("The gamma tensor doesn't have the right shape", nameof(gamma));
            if (!beta.MatchShape(gamma)) throw new ArgumentException("The beta tensor doesn't have the right shape", nameof(beta));
            if (!x.MatchShape(y)) throw new ArgumentException("The input and output tensors must have the same shape", nameof(y));
            switch (mode)
            {
                // A single mu and variance value per input channel
                case NormalizationMode.Spatial:
                    BatchNormalizationForward(info, x, mu, sigma2, gamma, beta, y);
                    break;

                // Each individual activation has its own median and variance
                case NormalizationMode.PerActivation:
                    BatchNormalizationForward(x, mu, sigma2, gamma, beta, y);
                    break;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Invalid normalization mode");
            }
        }

        /// <summary>
        /// Executes the backward pass through a batch normalization layer
        /// </summary>
        /// <param name="mode">The desired normalization mode to apply</param>
        /// <param name="info">The ifo on the input <see cref="Tensor"/> to process</param>
        /// <param name="x">The input <see cref="Tensor"/> to normalize</param>
        /// <param name="mu">A <see cref="Tensor"/> with the temporary median values calculated in the forward pass</param>
        /// <param name="sigma2">A <see cref="Tensor"/> with the temporary standard deviation values calculated in the forward pass</param>
        /// <param name="gamma">The layer gamma parameters</param>
        /// <param name="dy">The output error delta <see cref="Tensor"/></param>
        /// <param name="dx">The resulting backpropagated error delta <see cref="Tensor"/></param>
        public static void BatchNormalizationBackwardData(
            NormalizationMode mode, in TensorInfo info, in Tensor x, 
            in Tensor mu, in Tensor sigma2, in Tensor gamma, 
            in Tensor dy, in Tensor dx)
        {
            // Checks
            if (!sigma2.MatchShape(mu)) throw new ArgumentException("Invalid standard deviation tensor shape", nameof(sigma2));
            if (!gamma.MatchShape(sigma2)) throw new ArgumentException("The gamma tensor doesn't have the right shape", nameof(gamma));
            if (!x.MatchShape(dy)) throw new ArgumentException("The input and output tensors must have the same shape", nameof(dy));
            if (!x.MatchShape(dx)) throw new ArgumentException("The input the resulting error tensor must have the same shape", nameof(dx));
            switch (mode)
            {
                case NormalizationMode.Spatial:
                    BatchNormalizationBackwardData(info, x, mu, sigma2, gamma, dy, dx);
                    break;
                case NormalizationMode.PerActivation:
                    BatchNormalizationBackwardData(x, mu, sigma2, gamma, dy, dx);
                    break;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Invalid normalization mode");
            }
        }

        /// <summary>
        /// Calculates the gradient with respect to the gamma <see cref="Tensor"/> in a batch normalization layer
        /// </summary>
        /// <param name="mode">The desired normalization mode to apply</param>
        /// <param name="info">The ifo on the input <see cref="Tensor"/> to process</param>
        /// <param name="x">The input <see cref="Tensor"/> used in the forward pass</param>
        /// <param name="mu">A <see cref="Tensor"/> with the temporary median values calculated in the forward pass</param>
        /// <param name="sigma2">A <see cref="Tensor"/> with the temporary standard deviation values calculated in the forward pass</param>
        /// <param name="dy">The output <see cref="Tensor"/> error delta for the current layer</param>
        /// <param name="dgamma">The resulting gamma gradient <see cref="Tensor"/></param>
        public static void BatchNormalizationBackwardGamma(
            NormalizationMode mode, in TensorInfo info, in Tensor x, 
            in Tensor mu, in Tensor sigma2, 
            in Tensor dy, in Tensor dgamma)
        {
            // Checks
            if (!sigma2.MatchShape(mu)) throw new ArgumentException("Invalid standard deviation tensor shape", nameof(sigma2));
            if (!dgamma.MatchShape(sigma2)) throw new ArgumentException("Invalid gamma gradient tensor size", nameof(dgamma));
            if (!x.MatchShape(dy)) throw new ArgumentException("The input and output tensors must have the same shape", nameof(dy));
            switch (mode)
            {
                case NormalizationMode.Spatial:
                    BatchNormalizationBackwardGamma(info, x, mu, sigma2, dy, dgamma);
                    break;
                case NormalizationMode.PerActivation:
                    BatchNormalizationBackwardGamma(x, mu, sigma2, dy, dgamma);
                    break;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Invalid normalization mode");
            }
        }

        /// <summary>
        /// Calculates the gradient with respect to the beta <see cref="Tensor"/> in a batch normalization layer
        /// </summary>
        /// <param name="mode">The desired normalization mode to apply</param>
        /// <param name="info">The ifo on the input <see cref="Tensor"/> to process</param>
        /// <param name="dy">The output <see cref="Tensor"/> error delta for the current layer</param>
        /// <param name="dbeta">The resulting beta gradient <see cref="Tensor"/></param>
        public static void BatchNormalizationBackwardBeta(
            NormalizationMode mode, in TensorInfo info, in Tensor dy, in Tensor dbeta)
        {
            if (info.Size != dy.Length) throw new ArgumentException("The tensor shape doesn't match the input info", nameof(dy));
            switch (mode)
            {
                case NormalizationMode.Spatial:
                    BatchNormalizationBackwardBeta(info, dy, dbeta);
                    break;
                case NormalizationMode.PerActivation: 
                    if (!dbeta.MatchShape(1, dy.Length)) throw new ArgumentException("The beta tensor must have a value for output feature", nameof(dbeta));
                    FullyConnectedBackwardBias(dy, dbeta); // Vertical compression
                    break;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Invalid normalization mode");
            }
        }

        #region Spatial

        // Spatial forward training batch normalization
        private static unsafe void BatchNormalizationForward(
            in TensorInfo info, in Tensor x, 
            float factor, in Tensor mu, in Tensor sigma2, 
            in Tensor gamma, in Tensor beta, in Tensor y)
        {
            // Setup
            if (!mu.MatchShape(1, info.Channels)) throw new ArgumentException("Invalid mu tensor size");
            int
                n = x.Entities,
                l = x.Length,
                nhw = x.Entities * info.SliceSize,
                slice = info.SliceSize;
            float* px = x, pmu = mu, psigma2 = sigma2, py = y, pg = gamma, pb = beta;

            // Mean and variance
            Parallel.For(0, info.Channels, c =>
            {
                // Mu
                float mc = 0;
                float* start = px + slice * c;
                for (int i = 0; i < n; i++)
                {
                    float* offset = start + i * l;
                    for (int xy = 0; xy < slice; xy++)
                        mc += offset[xy];
                }
                pmu[c] = mc /= nhw * factor + pmu[c] * (1 - factor);

                // Variance
                float sc = 0;
                for (int i = 0; i < n; i++)
                {
                    float* offset = start + i * l;
                    for (int xy = 0; xy < slice; xy++)
                    {
                        float sq = offset[xy] - mc;
                        sc += sq * sq;
                    }
                }
                psigma2[c] = sc / nhw * factor + psigma2[c] * (1 - factor);

            }).AssertCompleted();

            // Normalization
            Parallel.For(0, info.Channels, c =>
            {
                float
                    gc = pg[c],
                    bc = pb[c],
                    mc = pmu[c],
                    sqrt_1 = 1 / (float)Math.Sqrt(psigma2[c] + CUDNN_BN_MIN_EPSILON);
                float*
                    start = px + slice * c,
                    end = py + slice * c;
                for (int i = 0; i < n; i++)
                {
                    float*
                        offset = start + i * l,
                        target = end + i * l;
                    for (int xy = 0; xy < slice; xy++)
                    {
                        float hat = (offset[xy] - mc) * sqrt_1;
                        target[xy] = gc * hat + bc;
                    }
                }
            }).AssertCompleted();
        }

        // Spatial forward inference batch normalization
        private static unsafe void BatchNormalizationForward(
            in TensorInfo info, in Tensor x, 
            in Tensor mu, in Tensor sigma2, 
            in Tensor gamma, in Tensor beta, in Tensor y)
        {
            // Setup
            if (!mu.MatchShape(1, info.Channels)) throw new ArgumentException("Invalid mu tensor size");
            int
                n = x.Entities,
                l = x.Length,
                slice = info.SliceSize;
            float* px = x, pmu = mu, psigma2 = sigma2, py = y, pg = gamma, pb = beta;
            Parallel.For(0, info.Channels, c =>
            {
                float
                    gc = pg[c],
                    bc = pb[c],
                    mc = pmu[c],
                    sqrt_1 = 1 / (float)Math.Sqrt(psigma2[c] + CUDNN_BN_MIN_EPSILON);
                float*
                    start = px + slice * c,
                    end = py + slice * c;
                for (int i = 0; i < n; i++)
                {
                    float*
                        offset = start + i * l,
                        target = end + i * l;
                    for (int xy = 0; xy < slice; xy++)
                    {
                        float hat = (offset[xy] - mc) * sqrt_1;
                        target[xy] = gc * hat + bc;
                    }
                }
            }).AssertCompleted();
        }

        // Spatial backward batch normalization
        private static unsafe void BatchNormalizationBackwardData(
            in TensorInfo info, in Tensor x, 
            in Tensor mu, in Tensor sigma2, in Tensor gamma, 
            in Tensor dy, in Tensor dx)
        {
            if (!mu.MatchShape(1, info.Channels)) throw new ArgumentException("Invalid mu tensor size");
            int
                n = dx.Entities,
                l = dx.Length,
                nhw = x.Entities * info.SliceSize,
                slice = info.SliceSize;
            float* px = x, pg = gamma, pmu = mu, psigma2 = sigma2, pdy = dy, pdx = dx;
            Parallel.For(0, info.Channels, c =>
            {
                // Calculate the two summatories
                float
                    mc = pmu[c],
                    sc = psigma2[c],
                    left = 1f / nhw * pg[c] / (float)Math.Sqrt(psigma2[c] + CUDNN_BN_MIN_EPSILON),
                    _2nd = 0,
                    _3rdRight = 0;
                float*
                    startdy = pdy + slice * c,
                    startx = px + slice * c;
                for (int i = 0; i < n; i++, startdy += l, startx += l)
                for (int xy = 0; xy < slice; xy++)
                {
                    float pdyicxy = startdy[xy];
                    _2nd += pdyicxy;
                    _3rdRight += pdyicxy * (startx[xy] - mc);
                }

                // Assign the backpropagated tensor
                float* startdx = pdx + slice * c;
                startdy = pdy + slice * c;
                startx = px + slice * c;
                for (int i = 0; i < n; i++, startdy += l, startx += l, startdx += l)
                for (int xy = 0; xy < slice; xy++)
                    startdx[xy] = left * (nhw * startdy[xy] - _2nd - (startx[xy] - mc) / (sc + CUDNN_BN_MIN_EPSILON) * _3rdRight);

            }).AssertCompleted();
        }

        // Spatial batch normalization gamma gradient
        private static unsafe void BatchNormalizationBackwardGamma(
            in TensorInfo info, in Tensor x, 
            in Tensor mu, in Tensor sigma2, 
            in Tensor dy, in Tensor dgamma)
        {
            if (!mu.MatchShape(1, info.Channels)) throw new ArgumentException("Invalid mu tensor size");
            int
                n = x.Entities,
                l = x.Length,
                slice = info.SliceSize;
            float* px = x, pdy = dy, pdg = dgamma, pmu = mu, psigma2 = sigma2;
            Parallel.For(0, info.Channels, c =>
            {
                float gc = 0, mc = pmu[c], sc = (float)Math.Sqrt(psigma2[c] + CUDNN_BN_MIN_EPSILON);
                int offset = slice * c;
                for (int i = 0; i < n; i++, offset += l)
                for (int xy = 0; xy < slice; xy++)
                    gc += pdy[offset + xy] * (px[offset + xy] - mc) / sc;
                pdg[c] = gc;
            }).AssertCompleted();
        }

        // Spatial batch normalization beta gradient
        private static unsafe void BatchNormalizationBackwardBeta(in TensorInfo info, in Tensor dy, in Tensor dbeta)
        {
            // Setup
            if (!dbeta.MatchShape(1, info.Channels)) throw new ArgumentException("The beta tensor must have a value for each input channel", nameof(dbeta));
            int
                n = dy.Entities,
                slice = info.SliceSize,
                l = info.Size;
            float* pdy = dy, pdbeta = dbeta;

            // Accumulate the output gradient
            Parallel.For(0, info.Channels, c =>
            {
                float bc = 0;
                float* start = pdy + c * slice;
                for (int i = 0; i < n; i++, start += l)
                for (int xy = 0; xy < slice; xy++)
                    bc += start[xy];
                pdbeta[c] = bc;
            }).AssertCompleted();
        }

        #endregion

        #region Per activation

        // Per-activation forward training batch normalization
        private static unsafe void BatchNormalizationForward(
            in Tensor x,
            float factor, in Tensor mu, in Tensor sigma2, 
            in Tensor gamma, in Tensor beta, in Tensor y)
        {
            if (!mu.MatchShape(1, x.Length)) throw new ArgumentException("Invalid mu tensor size");
            int
                n = x.Entities,
                l = x.Length;
            float* px = x, pmu = mu, psigma2 = sigma2, py = y, pg = gamma, pb = beta;
            Parallel.For(0, l, j =>
            {
                // Mean
                float mi = 0;
                for (int i = 0; i < n; i++)
                    mi += px[i * l + j];
                pmu[j] = mi /= n * factor + pmu[j] * (1 - factor);

                // Variance
                float sl = 0;
                for (int i = 0; i < n; i++)
                {
                    float hm = px[i * l + j] - mi;
                    sl += hm * hm;
                }
                psigma2[j] = sl / n * factor + psigma2[j] * (1 - factor);

            }).AssertCompleted();

            // Apply the batch normalization pass
            Parallel.For(0, n, i =>
            {
                int offset = i * l;
                for (int j = 0; j < l; j++)
                {
                    float hat = (px[offset + j] - pmu[j]) / (float)Math.Sqrt(psigma2[j] + CUDNN_BN_MIN_EPSILON);
                    py[offset + j] = pg[j] * hat + pb[j];
                }
            }).AssertCompleted();
        }

        // Per-activation forward inference batch normalization
        private static unsafe void BatchNormalizationForward(
            in Tensor x,
            in Tensor mu, in Tensor sigma2, 
            in Tensor gamma, in Tensor beta, in Tensor y)
        {
            if (!mu.MatchShape(1, x.Length)) throw new ArgumentException("Invalid mu tensor size");
            int
                n = x.Entities,
                l = x.Length;
            float* px = x, pmu = mu, psigma2 = sigma2, py = y, pg = gamma, pb = beta;
            Parallel.For(0, n, i =>
            {
                int offset = i * l;
                for (int j = 0; j < l; j++)
                {
                    float hat = (px[offset + j] - pmu[j]) / (float)Math.Sqrt(psigma2[j] + CUDNN_BN_MIN_EPSILON);
                    py[offset + j] = pg[j] * hat + pb[j];
                }
            }).AssertCompleted();
        }

        // Per-activation backward batch normalization
        private static unsafe void BatchNormalizationBackwardData(
            in Tensor x, 
            in Tensor mu, in Tensor sigma2, in Tensor gamma, 
            in Tensor dy, in Tensor dx)
        {
            if (!mu.MatchShape(1, x.Length)) throw new ArgumentException("Invalid mu tensor size");
            int
                n = dx.Entities,
                l = dx.Length;
            float* px = x, pg = gamma, pmu = mu, psigma2 = sigma2, pdy = dy, pdx = dx;
            Parallel.For(0, n, i =>
            {
                for (int j = 0; j < l; j++)
                {
                    float
                        left = 1f / n * pg[j] / (float)Math.Sqrt(psigma2[j] + CUDNN_BN_MIN_EPSILON),
                        _1st = n * pdy[i * l + j],
                        _2nd = 0,
                        _3rdLeft = (px[i * l + j] - pmu[j]) / (psigma2[j] + CUDNN_BN_MIN_EPSILON),
                        _3rdRight = 0;
                    for (int k = 0; k < n; k++)
                    {
                        float pdykj = pdy[k * l + j];
                        _2nd += pdykj;
                        _3rdRight += pdykj * (px[k * l + j] - pmu[j]);
                    }
                    pdx[i * l + j] = left * (_1st - _2nd - _3rdLeft * _3rdRight);
                }
            }).AssertCompleted();
        }

        // Per-activation batch normalization gamma gradient
        private static unsafe void BatchNormalizationBackwardGamma(
            in Tensor x, 
            in Tensor mu, in Tensor sigma2, 
            in Tensor dy, in Tensor dgamma)
        {
            if (!mu.MatchShape(1, x.Length)) throw new ArgumentException("Invalid mu tensor size");
            int
                n = x.Entities,
                l = x.Length;
            float* px = x, pdy = dy, pdg = dgamma, pmu = mu, psigma2 = sigma2;
            Parallel.For(0, x.Length, j =>
            {
                float sum = 0, sj = (float)Math.Sqrt(psigma2[j] + CUDNN_BN_MIN_EPSILON);
                for (int i = 0; i < n; i++)
                {
                    float hat = (px[i * l + j] - pmu[j]) / sj;
                    sum += pdy[i * l + j] * hat;
                }
                pdg[j] = sum;
            }).AssertCompleted();
        }

        #endregion
    }
}
