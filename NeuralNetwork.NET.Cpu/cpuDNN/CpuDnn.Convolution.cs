using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkDotNet.Core.APIs.Models;
using NeuralNetworkDotNet.Core.Helpers;

namespace NeuralNetworkDotNet.Cpu.cpuDNN
{
    /// <summary>
    /// A <see langword="class"/> that contains primitives to implement a neural network running on CPU
    /// </summary>
    public static partial class CpuDnn
    {
        /// <summary>
        /// Performs a forward convolution operation for a network layer
        /// </summary>
        /// <param name="x">The source <see cref="Tensor"/>, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="w">The list of convolution kernels to apply to each image</param>
        /// <param name="b">The bias <see cref="Tensor"/> to sum to the resulting images</param>
        /// <param name="y">The resulting convolution <see cref="Tensor"/></param>
        /// <exception cref="ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        public static void ConvolutionForward([NotNull] Tensor x, [NotNull] Tensor w, [NotNull] Tensor b, [NotNull] Tensor y)
        {
            Guard.IsFalse(w.CHW == 0, nameof(w), "The kernels can't be empty");
            Guard.IsFalse(x.HW < w.HW, "Each subdivided tensor must at least have the size of the kernels");
            Guard.IsTrue(x.C == w.C, "The depth of each kernel must be equal to the depth of each input volume");
            Guard.IsTrue(b.CHW == w.N, "The sum vector must be as long as the depth of the input volume");

            /* ============================
             * Valid convolution (forward)
             * ============================
             * Input volume:    H * W * sourceDepth (for each sample)
             * Kernels:         HK * WK * sourceDepth * kernelsDepth (same depth as the input, each kernel is a 3D volume)
             * Output:          kernelsDepth slices, one for each 3D kernel used */
            int
                nKernels = w.N,
                kw = w.CHW,
                kSize = w.HW,
                kHeight = w.H,
                kWidth = w.W,
                n = x.N,
                l = x.CHW,
                sourceDepth = x.C,
                imgSize = x.HW,
                imgHeight = x.H,
                imgWidth = x.W,
                hResult = imgHeight - kHeight + 1,              // Size of each image edge after the convolution
                wResult = imgWidth - kWidth + 1,
                convolutionOutputSize = hResult * wResult,      // Size of each processed image
                finalWidth = convolutionOutputSize * nKernels;  // Final size of each sample row

            Guard.IsTrue(y.Shape == (n, nKernels, hResult, wResult), nameof(y), "Invalid output tensor shape");

            // Process the valid convolution
            void ForwardKernel(int index)
            {
                // Calculate the current indexes
                int
                    iSample = index / nKernels, // Sample index
                    k = index % nKernels,       // Kernel index
                    targetBaseOffset = iSample * finalWidth + k * convolutionOutputSize,
                    sourceBaseOffset = iSample * l,
                    kernelBaseOffset = k * kw;

                ref var rx = ref x.Span.GetPinnableReference();
                ref var rw = ref w.Span.GetPinnableReference();
                ref var rb = ref b.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                // Process the current convolution slice
                for (var i = 0; i < hResult; i++)
                {
                    int
                        targetRowOffset = targetBaseOffset + i * hResult,
                        xEnd = i + kHeight - 1;
                    for (var j = 0; j < wResult; j++)
                    {
                        var highY = j + kWidth - 1;
                        var temp = 0f;
                        for (var z = 0; z < sourceDepth; z++)
                        {
                            int
                                sourceDepthOffset = sourceBaseOffset + z * imgSize,
                                kernelDepthOffset = kernelBaseOffset + z * kSize;
                            for (var r = i; r <= xEnd; r++)
                            {
                                int
                                    sourceRowOffset = sourceDepthOffset + r * imgWidth,
                                    kernelRowOffset = kernelDepthOffset + (xEnd - r) * kWidth + highY;
                                for (var c = j; c <= highY; c++)
                                {
                                    temp += Unsafe.Add(ref rx, sourceRowOffset + c) * Unsafe.Add(ref rw, kernelRowOffset - c);
                                }
                            }
                        }

                        Unsafe.Add(ref ry, targetRowOffset + j) = temp + Unsafe.Add(ref rb, k);
                    }
                }
            }

            Parallel.For(0, n * nKernels, ForwardKernel);
        }

        /// <summary>
        /// Performs the backwards pass on a convolutional layer
        /// </summary>
        /// <param name="dy">The output error <see cref="Tensor"/></param>
        /// <param name="w">The layer convolution kernels</param>
        /// <param name="dx">The resulting backpropagated error <see cref="Tensor"/></param>
        /// <exception cref="ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        [SuppressMessage("ReSharper", "AccessToDisposedClosure")]
        public static void ConvolutionBackwardData([NotNull] Tensor dy, [NotNull] Tensor w, [NotNull] Tensor dx)
        {
            Guard.IsTrue(dy.HW * dy.C == dy.CHW, nameof(dy), "Invalid depth parameter for the input tensor");
            Guard.IsFalse(dy.HW < w.HW, "Each subdivided tensor must at least have the size of the kernels");
            Guard.IsTrue(dy.C == w.N, "The source depth must be equal to the number of kernels");

            /* ============================
             * Full convolution (backwards)
             * ============================
             * Input volume:    H*W*sourceDepth (the delta(l + 1) for each sample)
             * Kernels:         HK*WK*kernelsDepth*sourceDepth (a kernel for each input slice)
             * Output:          kernelsDepth slices, each is the sum of the i-th slice of all the kernelsDepth kernels with convoluted with the i-th input slice */
            int
                nKernels = w.N,
                kw = w.CHW,
                kSize = w.HW,
                kHeight = w.H,
                kWidth = w.W,
                kDepth = w.C,
                n = dy.N,
                l = dy.CHW,
                imgSize = dy.HW,
                imgHeight = dy.H,
                imgWidth = dy.W,
                hResult = imgHeight + kHeight - 1,                  // Size of each image edge after the convolution
                wResult = imgWidth + kWidth - 1,
                convolutionOutputSize = hResult * wResult,          // Size of each processed image
                finalWidth = convolutionOutputSize * kDepth;        // Final size of each sample row

            // Rotate the layer kernels
            using (var w180 = Tensor.Like(w))
            {
                Rotate180(w, w180);

                // Process the full convolution
                void BackwardsKernel(int index)
                {
                    // Calculate the current indexes
                    int
                        iSample = index / kDepth,         // Sample index
                        iKernelDepth = index % kDepth,    // Kernel index
                        targetBaseOffset = iSample * finalWidth + iKernelDepth * convolutionOutputSize,
                        sourceBaseOffset = iSample * l,
                        kernelBaseOffset = iKernelDepth * kSize;

                    ref var rdy = ref dy.Span.GetPinnableReference();
                    ref var rw180 = ref w180.Span.GetPinnableReference();
                    ref var rdx = ref dx.Span.GetPinnableReference();

                    // Process the convolution slice
                    for (var i = 0; i < hResult; ++i)
                    {
                        int
                            lowX = Math.Max(0, i - kHeight + 1),
                            highX = Math.Min(imgHeight - 1, i),
                            targetRowOffset = targetBaseOffset + i * hResult;
                        for (var j = 0; j < hResult; ++j)
                        {
                            int
                                lowY = Math.Max(0, j - kWidth + 1),
                                highY = Math.Min(imgWidth - 1, j);
                            var temp = 0f;
                            for (var z = 0; z < nKernels; z++)
                            {
                                int
                                    sourceDepthOffset = sourceBaseOffset + z * imgSize,
                                    kernelDepthOffset = kernelBaseOffset + z * kw;
                                for (var x = lowX; x <= highX; ++x)
                                {
                                    int
                                        sourceRowOffset = sourceDepthOffset + x * imgWidth,
                                        kernelRowOffset = kernelDepthOffset + (i - x) * kWidth + j;
                                    for (var y = lowY; y <= highY; ++y)
                                    {
                                        temp += Unsafe.Add(ref rdy, sourceRowOffset + y) * Unsafe.Add(ref rw180, kernelRowOffset - y);
                                    }
                                }
                            }

                            Unsafe.Add(ref rdx, targetRowOffset + j) = temp;
                        }
                    }
                }

                Parallel.For(0, n * kDepth, BackwardsKernel);
            }
        }

        /// <summary>
        /// Performs a the backward convolution operation for a network layer and computes the gradient with respect to the layer weights
        /// </summary>
        /// <param name="x">The source <see cref="Tensor"/>, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="dy">The output error <see cref="Tensor"/></param>
        /// <param name="dw">The resulting weights gradient</param>
        /// <exception cref="ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        [SuppressMessage("ReSharper", "AccessToDisposedClosure")]
        public static void ConvolutionBackwardFilter([NotNull] Tensor x, [NotNull] Tensor dy, [NotNull] Tensor dw)
        {
            Guard.IsFalse(x.HW < dy.HW, "Each subdivided tensor must at least have the size of the kernels");
            Guard.IsTrue(dy.N == x.N, nameof(dy), "There must be a delta volume for each activation sample");

            /* ============================
             * Valid convolution (gradient)
             * ============================
             * Input volume:    H*W*sourceDepth (for each sample)
             * Kernels:         HK*WK*sourceDepth*kernelsDepth (delta(l + 1) used to calculate the 3D gradient for each kernel)
             * Output:          sourceDepth*kernelsDepth slices, where each stack of sourceDepth slices is the gradient for the i-th kernel */

            int
                kw = dy.CHW,
                kDepth = dy.C,
                kSize = dy.HW,
                kHeight = dy.H,
                kWidth = dy.W,
                n = x.N,
                l = x.CHW,
                imgSize = x.HW,
                imgHeight = x.H,
                imgWidth = x.W,
                hResult = imgHeight - kHeight + 1,              // Size of each image edge after the convolution
                wResult = imgWidth - kWidth + 1,
                convolutionOutputSize = hResult * wResult,      // Size of each processed image
                gradientSize = convolutionOutputSize * x.C,     // Size of each calculated gradient (one for each original kernel, so for each input delta)
                finalWidth = gradientSize * dy.C,               // Final size of each sample row
                iterationsPerSample = x.C * kDepth;             // Each sample has its own list of 3D gradients, one for each kernel

            // Rotate the inputs and prepare the temporary tensor
            using (var xt = Tensor.Like(x))
            using (var dwTemp = Tensor.New(x.N, finalWidth))
            {
                Rotate180(x, xt);

                // Process the valid convolution
                void GradientKernel(int index)
                {
                    // Calculate the current indexes
                    int
                        iSample = index / iterationsPerSample,      // Sample index
                        iMod = index % iterationsPerSample,
                        iSampleDepth = iMod / kDepth,               // Depth of the current gradient
                        iKernelDepth = iMod % kDepth,               // Output gradient index
                        sourceBaseOffset = iSample * l + iSampleDepth * imgSize,
                        kernelBaseOffset = iSample * kw + iKernelDepth * kSize,
                        resultBaseOffset = iSample * finalWidth + iKernelDepth * gradientSize + iSampleDepth * convolutionOutputSize;

                    ref var rdy = ref dy.Span.GetPinnableReference();
                    ref var rx = ref x.Span.GetPinnableReference();
                    ref var rdw = ref dwTemp.Span.GetPinnableReference();

                    for (var i = 0; i < hResult; i++)
                    {
                        int
                            targetRowOffset = resultBaseOffset + i * hResult,
                            xEnd = i + kHeight - 1;
                        for (var j = 0; j < hResult; j++)
                        {
                            var highY = j + kWidth - 1;
                            var temp = 0f;
                            for (var r = i; r <= xEnd; r++)
                            {
                                int
                                    sourceRowOffset = sourceBaseOffset + r * imgWidth,
                                    kernelRowOffset = kernelBaseOffset + (xEnd - r) * kWidth + highY;
                                for (var c = j; c <= highY; c++)
                                {
                                    temp += Unsafe.Add(ref rx, sourceRowOffset + c) * Unsafe.Add(ref rdy, kernelRowOffset - c);
                                }
                            }

                            Unsafe.Add(ref rdw, targetRowOffset + j) = temp;
                        }
                    }
                }

                Parallel.For(0, n * iterationsPerSample, GradientKernel);

                /* ==========================
                 * Gradient compression
                 * ==========================
                 * At this point, the temporary tensor has the series of (p,q) gradients for all the layer
                 * kernels, where p is the input depth and q is the kernel index.
                 * The final weights gradient is the sum for all the samples in the current training batch */
                Tensor wPlane = dw.Reshape(1, dw.NCHW);  // The gradient is [q,p]-shaped, flatten to the size of each sample before compressing
                CompressVertically(dwTemp, wPlane);
            }
        }

        /// <summary>
        /// Performs a the backward convolution operation for a network layer and computes the gradient with respect to the layer biases
        /// </summary>
        /// <param name="dy">The output error <see cref="Tensor"/></param>
        /// <param name="db">The resulting gradient</param>
        /// <exception cref="ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        [PublicAPI]
        [SuppressMessage("ReSharper", "AccessToDisposedClosure")]
        public static void ConvolutionBackwardBias([NotNull] Tensor dy, [NotNull] Tensor db)
        {
            Guard.IsTrue(dy.H == dy.W, nameof(dy), "The input images must be squares");

            int
                depth = dy.C,
                h = dy.N,
                w = dy.CHW,
                imgSize = dy.HW;

            using (var temp = Tensor.New(h, depth))
            {
                // Kernel to sum each slice
                void Kernel(int index)
                {
                    // Calculate the current indexes
                    int
                        iSample = index / depth,    // Sample index
                        z = index % depth,          // 2D slice index
                        baseOffset = iSample * w + z * imgSize;
                    var sum = 0f;

                    ref var rdy = ref dy.Span.GetPinnableReference();
                    ref var rt = ref temp.Span.GetPinnableReference();

                    for (var i = 0; i < imgSize; i++)
                    {
                        sum += Unsafe.Add(ref rdy, baseOffset + i);
                    }

                    Unsafe.Add(ref rt, iSample * depth + z) = sum;
                }

                Parallel.For(0, h * depth, Kernel);
                CompressVertically(temp, db);
            }
        }

        #region Tools

        /// <summary>
        /// Rotates the input <see cref="Tensor"/> by 180 degrees along the C axis
        /// </summary>
        /// <param name="x">The input <see cref="Tensor"/> to rotate</param>
        /// <param name="y">The rotated input <see cref="Tensor"/> to hold the results</param>
        private static void Rotate180([NotNull] Tensor x, [NotNull] Tensor y)
        {
            Guard.IsTrue(x.Shape == y.Shape, "The output tensor doesn't match the shape of the input");
            Guard.IsTrue(x.H == x.W, nameof(x), "The input images must be squares");

            int
                n = x.N,
                l = x.CHW,
                c = x.C,
                imgSize = x.HW,
                threshold = imgSize / 2,
                edge = imgSize - 1;
            bool odd = imgSize % 2 == 1;

            // Inversion kernel
            void Kernel(int index)
            {
                // Calculate the current indexes
                int
                    iSample = index / c,    // Sample index
                    z = index % c,          // 2D slice index
                    baseOffset = iSample * l + z * imgSize;

                ref var rx = ref x.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                // Reverse the input tensor sequentially
                for (int i = 0; i < threshold; i++)
                {
                    int
                        left = baseOffset + i,
                        right = baseOffset + edge - i;
                    Unsafe.Add(ref ry, left) = Unsafe.Add(ref rx, right);
                    Unsafe.Add(ref ry, right) = Unsafe.Add(ref rx, left);
                }

                if (odd)
                {
                    int center = baseOffset + threshold;
                    Unsafe.Add(ref ry, center) = Unsafe.Add(ref rx, center);
                }
            }

            Parallel.For(0, n * c, Kernel);
        }

        /// <summary>
        /// Compresses a <see cref="Tensor"/> into a row by summing the components column by column
        /// </summary>
        /// <param name="x">The <see cref="Tensor"/> to compress</param>
        /// <param name="y">The resulting <see cref="Tensor"/></param>
        private static void CompressVertically([NotNull] Tensor x, [NotNull] Tensor y)
        {
            Guard.IsTrue((y.N, y.CHW) == (1, x.CHW), "The output tensor doesn't have the right shape");

            int
                n = x.N,
                l = x.CHW;

            void Kernel(int j)
            {
                var sum = 0f;

                ref var rx = ref x.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                for (var i = 0; i < n; i++)
                    sum += Unsafe.Add(ref rx, i * l + j);
                Unsafe.Add(ref ry, j) = sum;
            }

            Parallel.For(0, l, Kernel);
        }

        #endregion
    }
}
