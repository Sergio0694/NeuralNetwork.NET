using System;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.cpuDNN
{
    /// <inheritdoc cref="CpuDnn"/>
    public static partial class CpuDnn
    {
        #region Implementation

        /// <summary>
        /// Performs a forward convolution operation for a network layer
        /// </summary>
        /// <param name="x">The source <see cref="Tensor"/>, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="xInfo">The source volume info (depth and 2D slices size)</param>
        /// <param name="w">The list of convolution kernels to apply to each image</param>
        /// <param name="wInfo">The kernels volume info (depth and 2D slices size)</param>
        /// <param name="b">The bias <see cref="Tensor"/> to sum to the resulting images</param>
        /// <param name="y">The resulting convolution <see cref="Tensor"/></param>
        /// <exception cref="ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
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
            int
                n = x.Entities,
                l = x.Length,
                sourceDepth = xInfo.Channels,
                imgSize = xInfo.SliceSize,
                imgHeight = xInfo.Height,
                imgWidth = xInfo.Width;  // Size of an edge of one of the inner images per sample
            if (imgSize * xInfo.Channels != l) throw new ArgumentException("Invalid depth parameter for the input tensor", nameof(x));
            if (imgSize < kSize) throw new ArgumentException("Each subdivided tensor must at least have the size of the kernels");
            if (xInfo.Channels != wInfo.Channels) throw new ArgumentException("The depth of each kernel must be equal to the depth of each input volume");
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
            float* px = x, py = y, pw = w, pb = b;
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
                                    temp += px[sourceRowOffset + c] * pw[kernelRowOffset - c];
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
        /// Performs the backwards pass on a convolutional layer
        /// </summary>
        /// <param name="dy">The output error <see cref="Tensor"/></param>
        /// <param name="dyInfo">The info on the output <see cref="Tensor"/></param>
        /// <param name="w">The layer convolution kernels</param>
        /// <param name="wInfo">The kernels volume info (depth and 2D slices size)</param>
        /// <param name="dx">The resulting backpropagated error <see cref="Tensor"/></param>
        /// <param name="dxInfo">The info on the layer inputs</param>
        /// <exception cref="ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        public static unsafe void ConvolutionBackwardData(
            in Tensor dy, in TensorInfo dyInfo,
            in Tensor w, in TensorInfo wInfo,
            in Tensor dx, in TensorInfo dxInfo)
        {
            // Checks and local parameters
            int
                nKernels = w.Entities,
                kw = w.Length,
                kSize = kw / wInfo.Channels,
                kHeight = wInfo.Height,
                kWidth = wInfo.Width,
                kDepth = wInfo.Channels;
            int
                n = dy.Entities,
                l = dy.Length,
                imgSize = dyInfo.SliceSize,
                imgHeight = dyInfo.Height,
                imgWidth = dyInfo.Width;
            if (imgSize * dyInfo.Channels != l) throw new ArgumentException("Invalid depth parameter for the input tensor", nameof(dy));
            if (imgSize < kSize) throw new ArgumentException("Each subdivided tensor must at least have the size of the kernels");
            if (dyInfo.Channels != nKernels) throw new ArgumentException("The source depth must be equal to the number of kernels");

            // Rotate the layer kernels
            Rotate180(w, wInfo.Channels, out Tensor w180);

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
                            for (int x = lowX; x <= highX; ++x)
                            {
                                int
                                    sourceRowOffset = sourceDepthOffset + x * imgWidth,
                                    kernelRowOffset = kernelDepthOffset + (i - x) * kWidth + j;
                                for (int y = lowY; y <= highY; ++y)
                                {
                                    temp += pdy[sourceRowOffset + y] * pw180[kernelRowOffset - y];
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
        /// Performs a the backward convolution operation for a network layer and computes the gradient with respect to the layer weights
        /// </summary>
        /// <param name="x">The source <see cref="Tensor"/>, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="xInfo">The source volume info (depth and 2D slices size)</param>
        /// <param name="dy">The output error <see cref="Tensor"/></param>
        /// <param name="dyInfo">The output error volume info (depth and 2D slices size)</param>
        /// <param name="dw">The resulting weights gradient</param>
        /// <param name="wInfo">The info on the layer kernels</param>
        /// <exception cref="ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        public static unsafe void ConvolutionBackwardFilter(
            in Tensor x, in TensorInfo xInfo,
            in Tensor dy, in TensorInfo dyInfo,
            in Tensor dw, in TensorInfo wInfo)
        {
            // Checks and local parameters
            int
                nKernels = dy.Entities,
                kw = dy.Length,
                kDepth = dyInfo.Channels,
                kSize = kw / dyInfo.Channels,
                kHeight = dyInfo.Height,
                kWidth = dyInfo.Width;
            int
                n = x.Entities,
                l = x.Length,
                imgSize = xInfo.SliceSize,
                imgHeight = xInfo.Height,
                imgWidth = xInfo.Width;
            if (imgSize * xInfo.Channels != l) throw new ArgumentException("Invalid depth parameter for the input tensor", nameof(x));
            if (imgSize < kSize) throw new ArgumentOutOfRangeException(nameof(imgSize), "Each subdivided tensor must at least have the size of the kernels");
            if (nKernels != n) throw new ArgumentException("There must be a delta volume for each activation sample", nameof(dy));

            /* ============================
             * Valid convolution (gradient)
             * ============================
             * Input volume:    H*W*sourceDepth (for each sample)
             * Kernels:         HK*WK*sourceDepth*kernelsDepth (delta(l + 1) used to calculate the 3D gradient for each kernel)
             * Output:          sourceDepth*kernelsDepth slices, where each stack of sourceDepth slices is the gradient for the i-th kernel */
            int
                hResult = imgHeight - kHeight + 1,                          // Size of each image edge after the convolution
                wResult = imgWidth - kWidth + 1,
                convolutionOutputSize = hResult * wResult,                  // Size of each processed image
                gradientSize = convolutionOutputSize * xInfo.Channels,      // Size of each calculated gradient (one for each original kernel, so for each input delta)
                finalWidth = gradientSize * dyInfo.Channels,                // Final size of each sample row
                iterationsPerSample = xInfo.Channels * kDepth;              // Each sample has its own list of 3D gradients, one for each kernel

            // Rotate the inputs and prepare the temporary tensor
            Rotate180(x, xInfo.Channels, out Tensor xt);
            Tensor.New(x.Entities, finalWidth, out Tensor dwTemp);

            // Process the valid convolution
            float* px = xt, pdy = dy, pdw = dwTemp;
            void GradientKernel(int index)
            {
                // Calculate the current indexes
                int
                    iSample = index / iterationsPerSample,      // Sample index
                    iMod = index % iterationsPerSample,
                    iSampleDepth = iMod / kDepth,               // Depth of the current gradient
                    iKernelDepth = iMod % kDepth;               // Output gradient index

                // Process the current convolution slice
                int
                    sourceBaseOffset = iSample * l + iSampleDepth * imgSize,
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
                        for (int r = i; r <= xEnd; r++)
                        {
                            int
                                sourceRowOffset = sourceBaseOffset + r * imgWidth,
                                kernelRowOffset = kernelBaseOffset + (xEnd - r) * kWidth + highY;
                            for (int c = j; c <= highY; c++)
                            {
                                temp += px[sourceRowOffset + c] * pdy[kernelRowOffset - c];
                            }
                        }
                        pdw[targetRowOffset + j] = temp;
                    }
                }
            }
            Parallel.For(0, n * iterationsPerSample, GradientKernel).AssertCompleted();
            xt.Free();

            /* ==========================
             * Gradient compression
             * ==========================
             * At this point, the temporary tensor has the series of (p,q) gradients for all the layer
             * kernels, where p is the input depth and q is the kernel index.
             * The final weights gradient is the sum for all the samples in the current training batch */
            dw.Reshape(1, dw.Size, out Tensor wPlane);  // The gradient is [q,p]-shaped, flatten to the size of each sample before compressing
            CompressVertically(dwTemp, wPlane);
            dwTemp.Free();
        }

        /// <summary>
        /// Performs a the backward convolution operation for a network layer and computes the gradient with respect to the layer biases
        /// </summary>
        /// <param name="dy">The output error <see cref="Tensor"/></param>
        /// <param name="dyInfo">The info on the output <see cref="Tensor"/></param>
        /// <param name="db">The resulting gradient</param>
        /// <exception cref="ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        [PublicAPI]
        public static unsafe void ConvolutionBackwardBias(in Tensor dy, in TensorInfo dyInfo, in Tensor db)
        {
            // Checks and local parameters
            int
                depth = dyInfo.Channels,
                h = dy.Entities,
                w = dy.Length,
                imgSize = w % depth == 0 ? w / depth : throw new ArgumentException("Invalid depth parameter for the input tensor", nameof(dy)),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentException("The size of the input tensor isn't valid", nameof(dy));
            Tensor.New(h, depth, out Tensor temp);

            // Kernel to sum each slice
            float* ptemp = temp, psource = dy;
            void Kernel(int index)
            {
                // Calculate the current indexes
                int
                    iSample = index / depth,    // Sample index
                    z = index % depth;          // 2D slice index

                // Reverse the input tensor sequentially
                int baseOffset = iSample * w + z * imgSize;
                float sum = 0;
                for (int i = 0; i < imgSize; i++)
                {
                    sum += psource[baseOffset + i];
                }
                ptemp[iSample * depth + z] = sum;
            }
            Parallel.For(0, h * depth, Kernel).AssertCompleted();
            CompressVertically(temp, db);
            temp.Free();
        }

        #endregion

        #region Tools

        /// <summary>
        /// Rotates the input volume by 180 degrees
        /// </summary>
        /// <param name="x">The input <see cref="Tensor"/> to rotate</param>
        /// <param name="depth">The number of images per row</param>
        /// <param name="y">The rotated input <see cref="Tensor"/></param>
        private static unsafe void Rotate180(in Tensor x, int depth, out Tensor y)
        {
            // Checks and local parameters
            if (depth < 1) throw new ArgumentException("The number of images per row can't be lower than 1", nameof(depth));
            int
                n = x.Entities,
                l = x.Length,
                imgSize = l % depth == 0 ? l / depth : throw new ArgumentException("Invalid depth parameter for the input tensor", nameof(x)),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentException("The size of the input tensor isn't valid", nameof(x));
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

                // Reverse the input tensor sequentially
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

        /// <summary>
        /// Compresses a <see cref="Tensor"/> into a row by summing the components column by column
        /// </summary>
        /// <param name="x">The <see cref="Tensor"/> to compress</param>
        /// <param name="y">The resulting <see cref="Tensor"/></param>
        private static unsafe void CompressVertically(in Tensor x, in Tensor y)
        {
            // Preliminary checks and declarations
            int
                n = x.Entities,
                l = x.Length;
            if (!y.MatchShape(1, x.Length)) throw new ArgumentException("The output tensor doesn't have the right shape", nameof(y));
            float* px = x, py = y;

            // Compress the tensor
            void Kernel(int j)
            {
                float sum = 0;
                for (int i = 0; i < n; i++)
                    sum += px[i * l + j];
                py[j] = sum;
            }
            Parallel.For(0, l, Kernel).AssertCompleted();
        }

        #endregion
    }
}