using System;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.cpuDNN
{
    /// <summary>
    /// A static class with a collection of convolution extension methods
    /// </summary>
    public static partial class CpuDnn
    {
        /// <summary>
        /// Performs a forward convolution operation on the source matrix, using the given kernels
        /// </summary>
        /// <param name="x">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="xInfo">The source volume info (depth and 2D slices size)</param>
        /// <param name="w">The list of convolution kernels to apply to each image</param>
        /// <param name="wInfo">The kernels volume info (depth and 2D slices size)</param>
        /// <param name="biases">The bias vector to sum to the resulting images</param>
        /// <param name="result">The resulting convolution volume</param>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        public static unsafe void ConvolutionForward(
            in Tensor x, in TensorInfo xInfo,
            in Tensor w, in TensorInfo wInfo,
            in Tensor b,
            in Tensor y)
        {
            // Checks and local parameters
            if (w.Length == 0) throw new ArgumentException("The kernels can't be empty", nameof(w));
            int
                nKernels = w.Entities,
                kw = w.Length,
                kSize = kw / wInfo.Channels,
                kHeight = wInfo.Height,
                kWidth = wInfo.Width;
            if (kHeight < 2 || kWidth < 2) throw new ArgumentException("The kernel must be at least 2x2", nameof(w));
            int
                n = x.Entities,
                l = x.Length,
                sourceDepth = xInfo.Channels,
                imgSize = xInfo.SliceSize,
                imgHeight = xInfo.Height,
                imgWidth = xInfo.Width;  // Size of an edge of one of the inner images per sample
            if (imgSize * xInfo.Channels != l) throw new ArgumentException("Invalid depth parameter for the input matrix", nameof(x));
            if (imgSize < kSize) throw new ArgumentOutOfRangeException("Each subdivided matrix must at least have the size of the kernels");
            if (xInfo.Channels != wInfo.Channels) throw new InvalidOperationException("The depth of each kernel must be equal to the depth of each input volume");
            if (b.Length != nKernels) throw new ArgumentException("The sum vector must be as long as the depth of the input volume");

            /* ============================
             * Valid convolution (forward)
             * ============================
             * Input volume:    H*W*sourceDepth (for each sample)
             * Kernels:         HK*WK*sourceDepth*kernelsDepth (same depth as the input, each kernel is a 3D volume)
             * Output:          kernelsDepth slices, one for each 3D kernel used */
            int
                hResult = imgHeight - kHeight + 1,                  // Size of each image edge after the convolution
                wResult = imgWidth - kWidth + 1,
                convolutionOutputSize = hResult * wResult,          // Size of each processed image
                finalWidth = convolutionOutputSize * nKernels;      // Final size of each sample row
            if (!y.MatchShape(n, finalWidth)) throw new ArgumentException("Invalid output tensor shape", nameof(y));

            // Process the valid convolution
            float* psource = x, py = y, pkernels = w, pb = b;
            void ForwardKernel(int index)
            {
                // Calculate the current indexes
                int
                    iSample = index / nKernels,     // Sample index
                    k = index % nKernels;           // Kernel index

                // Process the current convolution slice
                int
                    targetBaseOffset = iSample * finalWidth + k * convolutionOutputSize,
                    sourceBaseOffset = iSample * l,
                    kernelBaseOffset = k * kw;
                for (int i = 0; i < hResult; i++)
                {
                    int
                        targetRowOffset = targetBaseOffset + i * hResult,
                        xEnd = i + kHeight - 1;
                    for (int j = 0; j < wResult; j++)
                    {
                        int highY = j + kWidth - 1;
                        float temp = 0.0f;
                        for (int z = 0; z < sourceDepth; z++)
                        {
                            int
                                sourceDepthOffset = sourceBaseOffset + z * imgSize,
                                kernelDepthOffset = kernelBaseOffset + z * kSize;
                            for (int r = i; r <= xEnd; r++)
                            {
                                int
                                    sourceRowOffset = sourceDepthOffset + r * imgWidth,
                                    kernelRowOffset = kernelDepthOffset + (xEnd - r) * kWidth + highY;
                                for (int c = j; c <= highY; c++)
                                {
                                    temp += psource[sourceRowOffset + c] * pkernels[kernelRowOffset - c];
                                }
                            }
                        }
                        py[targetRowOffset + j] = temp + pb[k];
                    }
                }
            }
            Parallel.For(0, n * nKernels, ForwardKernel).AssertCompleted();
        }

        /// <summary>
        /// Performs the full backwards convolution operation on the source matrix, using the given kernels
        /// </summary>
        /// <param name="dy">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="sourceInfo">The source volume info (depth and 2D slices size)</param>
        /// <param name="w">The list of convolution kernels to apply to each image</param>
        /// <param name="kernelsInfo">The kernels volume info (depth and 2D slices size)</param>
        /// <param name="dx">The resulting convolution volume</param>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        public static unsafe void ConvolutionBackwardData(
            in Tensor dy, in TensorInfo sourceInfo,
            in Tensor w, in TensorInfo kernelsInfo,
            in Tensor dx)
        {
            // Rotate the weights
            if (!w.MatchShape(sourceInfo.Channels, kernelsInfo.Size)) throw new ArgumentException("The input kernels don't have the right shape", nameof(w));
            Rotate180(w, sourceInfo.Channels, out Tensor w180);

            // Checks and local parameters
            int
                nKernels = w180.Entities,
                kw = w180.Length,
                kSize = kw / kernelsInfo.Channels,
                kHeight = kernelsInfo.Height,
                kWidth = kernelsInfo.Width,
                kDepth = kernelsInfo.Channels;
            if (kHeight < 2 || kWidth < 2) throw new ArgumentException(nameof(w), "The kernel must be at least 2x2");
            int
                n = dy.Entities,
                l = dy.Length,
                imgSize = sourceInfo.SliceSize,
                imgHeight = sourceInfo.Height,
                imgWidth = sourceInfo.Width;
            if (imgSize * sourceInfo.Channels != l) throw new ArgumentException(nameof(dy), "Invalid depth parameter for the input matrix");
            if (imgSize < kSize) throw new ArgumentOutOfRangeException("Each subdivided matrix must at least have the size of the kernels");
            if (sourceInfo.Channels != nKernels) throw new ArgumentException("The source depth must be equal to the number of kernels");

            /* ============================
             * Full convolution (backwards)
             * ============================
             * Input volume:    H*W*sourceDepth (the delta(l + 1) for each sample)
             * Kernels:         HK*WK*kernelsDepth*sourceDepth (a kernel for each input slice)
             * Output:          kernelsDepth slices, each is the sum of the i-th slice of all the kernelsDepth kernels with convoluted with the i-th input slice */
            int
                hResult = imgHeight + kHeight - 1,                  // Size of each image edge after the convolution
                wResult = imgWidth + kWidth - 1,
                convolutionOutputSize = hResult * wResult,          // Size of each processed image
                finalWidth = convolutionOutputSize * kDepth;        // Final size of each sample row
            if (!dx.MatchShape(n, finalWidth)) throw new ArgumentException("The output tensor doesn't have the right shape", nameof(dx));

            // Process the full convolution
            float* pdy = dy, pw180 = w180, pdx = dx;
            void BackwardsKernel(int index)
            {
                // Calculate the current indexes
                int
                    iSample = index / kDepth,         // Sample index
                    iKernelDepth = index % kDepth;    // Kernel index

                // Process the convolution slice
                int
                    targetBaseOffset = iSample * finalWidth + iKernelDepth * convolutionOutputSize,
                    sourceBaseOffset = iSample * l,
                    kernelBaseOffset = iKernelDepth * kSize;
                for (int i = 0; i < hResult; ++i)
                {
                    int
                        lowX = 0.Max(i - kHeight + 1),
                        highX = (imgHeight - 1).Min(i),
                        targetRowOffset = targetBaseOffset + i * hResult;
                    for (int j = 0; j < hResult; ++j)
                    {
                        int
                            lowY = 0.Max(j - kWidth + 1),
                            highY = (imgWidth - 1).Min(j);
                        float temp = 0.0f;
                        for (int z = 0; z < nKernels; z++)
                        {
                            int
                                sourceDepthOffset = sourceBaseOffset + z * imgSize,
                                kernelDepthOffset = kernelBaseOffset + z * kw;
                            for (int r = lowX; r <= highX; ++r)
                            {
                                int
                                    sourceRowOffset = sourceDepthOffset + r * imgWidth,
                                    kernelRowOffset = kernelDepthOffset + (i - r) * kWidth + j;
                                for (int c = lowY; c <= highY; ++c)
                                {
                                    temp += pdy[sourceRowOffset + c] * pw180[kernelRowOffset - c];
                                }
                            }
                        }
                        pdx[targetRowOffset + j] = temp;
                    }
                }
            }
            Parallel.For(0, n * kDepth, BackwardsKernel).AssertCompleted();

            w180.Free();
        }

        /// <summary>
        /// Performs a the gradient convolution operation on the source matrix, using the given kernels
        /// </summary>
        /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="sourceInfo">The source volume info (depth and 2D slices size)</param>
        /// <param name="kernels">The list of convolution kernels to apply to each image</param>
        /// <param name="kernelsInfo">The kernels volume info (depth and 2D slices size)</param>
        /// <param name="result">The resulting convolution volume</param>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        public static unsafe void ConvoluteGradient(
            in this Tensor source, in TensorInfo sourceInfo,
            in Tensor kernels, in TensorInfo kernelsInfo,
            out Tensor result)
        {
            // Checks and local parameters
            int
                nKernels = kernels.Entities,
                kw = kernels.Length,
                kDepth = kernelsInfo.Channels,
                kSize = kw / kernelsInfo.Channels,
                kHeight = kernelsInfo.Height,
                kWidth = kernelsInfo.Width;
            if (kHeight < 2 || kWidth < 2) throw new ArgumentException(nameof(kernels), "The kernel must be at least 2x2");
            int
                h = source.Entities,
                w = source.Length,
                imgSize = sourceInfo.SliceSize,
                imgHeight = sourceInfo.Height,
                imgWidth = sourceInfo.Width;
            if (imgSize * sourceInfo.Channels != w) throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix");
            if (imgSize < kSize) throw new ArgumentOutOfRangeException("Each subdivided matrix must at least have the size of the kernels");
            if (nKernels != h) throw new ArgumentException(nameof(kernels), "There must be a delta volume for each activation sample");

            /* ============================
             * Valid convolution (gradient)
             * ============================
             * Input volume:    H*W*sourceDepth (for each sample)
             * Kernels:         HK*WK*sourceDepth*kernelsDepth (delta(l + 1) used to calculate the 3D gradient for each kernel)
             * Output:          sourceDepth*kernelsDepth slices, where each stack of sourceDepth slices is the gradient for the i-th kernel */
            int
                hResult = imgHeight - kHeight + 1,                              // Size of each image edge after the convolution
                wResult = imgWidth - kWidth + 1,
                convolutionOutputSize = hResult * wResult,                      // Size of each processed image
                gradientSize = convolutionOutputSize * sourceInfo.Channels,     // Size of each calculated gradient (one for each original kernel, so for each input delta)
                finalWidth = gradientSize * kernelsInfo.Channels,               // Final size of each sample row
                iterationsPerSample = sourceInfo.Channels * kDepth;             // Each sample has its own list of 3D gradients, one for each kernel

            // Process the valid convolution
            Tensor.New(h, finalWidth, out result);
            float* psource = source, pkernels = kernels, presult = result;
            unsafe void GradientKernel(int index)
            {
                // Calculate the current indexes
                int
                    iSample = index / iterationsPerSample,      // Sample index
                    iMod = index % iterationsPerSample,
                    iSampleDepth = iMod / kDepth,               // Depth of the current gradient
                    iKernelDepth = iMod % kDepth;               // Output gradient index

                // Process the current convolution slice
                int
                    sourceBaseOffset = iSample * w + iSampleDepth * imgSize,
                    kernelBaseOffset = iSample * kw + iKernelDepth * kSize,
                    resultBaseOffset = iSample * finalWidth + iKernelDepth * gradientSize + iSampleDepth * convolutionOutputSize;
                for (int i = 0; i < hResult; i++)
                {
                    int
                        targetRowOffset = resultBaseOffset + i * hResult,
                        xEnd = i + kHeight - 1;
                    for (int j = 0; j < hResult; j++)
                    {
                        int highY = j + kWidth - 1;
                        float temp = 0.0f;
                        for (int x = i; x <= xEnd; x++)
                        {
                            int
                                sourceRowOffset = sourceBaseOffset + x * imgWidth,
                                kernelRowOffset = kernelBaseOffset + (xEnd - x) * kWidth + highY;
                            for (int y = j; y <= highY; y++)
                            {
                                temp += psource[sourceRowOffset + y] * pkernels[kernelRowOffset - y];
                            }
                        }
                        presult[targetRowOffset + j] = temp;
                    }
                }
            }
            Parallel.For(0, h * iterationsPerSample, GradientKernel).AssertCompleted();

            // TODO: correct gradient implementation
            throw new NotImplementedException("The CPU gradient convolution isn't implemented correctly yet");
        }

        #region Tools

        /// <summary>
        /// Compresses a convolution matrix into a row vector by summing each 2D slice in each row
        /// </summary>
        /// <param name="source">The matrix to compress</param>
        /// <param name="depth">The number of images per row</param>
        /// <param name="result">The resulting tensor</param>
        [PublicAPI]
        public static unsafe void CompressVertically(in this Tensor source, int depth, out Tensor result)
        {
            // Checks and local parameters
            if (depth < 1) throw new ArgumentOutOfRangeException(nameof(depth), "The number of images per row can't be lower than 1");
            int
                h = source.Entities,
                w = source.Length,
                imgSize = w % depth == 0 ? w / depth : throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(source), "The size of the input matrix isn't valid");
            Tensor.New(h, depth, out Tensor temp);

            // Kernel to sum each slice
            float* ptemp = temp, psource = source;
            void Kernel(int index)
            {
                // Calculate the current indexes
                int
                    iSample = index / depth,    // Sample index
                    z = index % depth;          // 2D slice index

                // Reverse the input matrix sequentially
                int baseOffset = iSample * w + z * imgSize;
                float sum = 0;
                for (int i = 0; i < imgSize; i++)
                {
                    sum += psource[baseOffset + i];
                }
                ptemp[iSample * depth + z] = sum;
            }
            Parallel.For(0, h * depth, Kernel).AssertCompleted();
            temp.CompressVertically(out result);
            temp.Free();
        }

        /// <summary>
        /// Rotates the input volume by 180 degrees
        /// </summary>
        /// <param name="x">The input matrix to rotate</param>
        /// <param name="depth">The number of images per row</param>
        /// <param name="y">The rotated input matrix</param>
        private static unsafe void Rotate180(in this Tensor x, int depth, out Tensor y)
        {
            // Checks and local parameters
            if (depth < 1) throw new ArgumentOutOfRangeException(nameof(depth), "The number of images per row can't be lower than 1");
            int
                n = x.Entities,
                l = x.Length,
                imgSize = l % depth == 0 ? l / depth : throw new ArgumentException(nameof(x), "Invalid depth parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(x), "The size of the input matrix isn't valid");
            int
                threshold = imgSize / 2,
                edge = imgSize - 1;
            bool odd = imgSize % 2 == 1;
            Tensor.New(n, l, out y);

            // Inversion kernel
            float* py = y, px = x;
            void Kernel(int index)
            {
                // Calculate the current indexes
                int
                    iSample = index / depth,    // Sample index
                    z = index % depth;          // 2D slice index

                // Reverse the input matrix sequentially
                int baseOffset = iSample * l + z * imgSize;
                for (int i = 0; i < threshold; i++)
                {
                    int
                        left = baseOffset + i,
                        right = baseOffset + edge - i;
                    py[left] = px[right];
                    py[right] = px[left];
                }
                if (odd)
                {
                    int center = baseOffset + threshold;
                    py[center] = px[center];
                }
            }
            Parallel.For(0, n * depth, Kernel).AssertCompleted();
        }

        #endregion
    }
}