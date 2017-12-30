using NeuralNetworkNET.APIs.Structs;
using System;
using System.Threading.Tasks;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.cpuDNN
{
    /// <inheritdoc cref="CpuDnn"/>
    public static partial class CpuDnn
    {
        /// <summary>
        /// Executes the forward pass on a max pooling layer with a 2x2 window and a stride of 2
        /// </summary>
        /// <param name="x">The input <see cref="Tensor"/> to pool</param>
        /// <param name="xInfo">The info on the input <see cref="Tensor"/></param>
        /// <param name="y">The resulting pooled <see cref="Tensor"/></param>
        /// <exception cref="ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        public static unsafe void PoolingForward(in Tensor x, in TensorInfo xInfo, in Tensor y)
        {
            int h = x.Entities, w = x.Length;
            if (h < 1 || w < 1) throw new ArgumentException("The input tensor isn't valid");
            int
                depth = xInfo.Channels,
                imgSize = w % depth == 0 ? w / depth : throw new ArgumentException("Invalid depth parameter for the input tensor", nameof(x)),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentException("The size of the input tensor isn't valid", nameof(x));
            int
                poolAxis = imgAxis / 2 + (imgAxis % 2 == 0 ? 0 : 1),
                poolSize = poolAxis * poolAxis,
                poolFinalWidth = depth * poolSize,
                edge = imgAxis - 1;
            if (!y.MatchShape(h, poolFinalWidth)) throw new ArgumentException("The output tensor shape isn't valid", nameof(y));

            // Pooling kernel
            float* px = x, py = y;
            void Kernel(int sample)
            {
                int
                    sourceBaseOffset = sample * w,
                    resultBaseOffset = sample * poolFinalWidth;
                for (int z = 0; z < depth; z++)
                {
                    int
                        sourceZOffset = sourceBaseOffset + z * imgSize,
                        resultZOffset = resultBaseOffset + z * poolSize,
                        c = 0;
                    for (int i = 0; i < imgAxis; i += 2)
                    {
                        int
                            sourceIOffset = sourceZOffset + i * imgAxis,
                            resultXOffset = resultZOffset + c * poolAxis,
                            r = 0;
                        if (i == edge)
                        {
                            // Last row
                            for (int j = 0; j < imgAxis; j += 2)
                            {
                                float max;
                                if (j == w - 1) max = px[sourceIOffset + j]; // Last column
                                else
                                {
                                    float
                                        left = px[sourceIOffset + j],
                                        right = px[sourceIOffset + j + 1];
                                    max = left > right ? left : right;
                                }
                                py[resultXOffset + r++] = max;
                            }
                        }
                        else
                        {
                            int sourceI_1Offset = sourceZOffset + (i + 1) * imgAxis;
                            for (int j = 0; j < imgAxis; j += 2)
                            {
                                float max;
                                if (j == edge)
                                {
                                    // Last column
                                    float
                                        up = px[sourceIOffset + j],
                                        down = px[sourceI_1Offset + j];
                                    max = up > down ? up : down;
                                }
                                else
                                {
                                    float
                                        upLeft = px[sourceIOffset + j],
                                        upRight = px[sourceIOffset + j + 1],
                                        downLeft = px[sourceI_1Offset + j],
                                        downRight = px[sourceI_1Offset + j + 1],
                                        maxUp = upLeft > upRight ? upLeft : upRight,
                                        maxDown = downLeft > downRight ? downLeft : downRight;
                                    max = maxUp > maxDown ? maxUp : maxDown;
                                }
                                py[resultXOffset + r++] = max;
                            }
                        }
                        c++;
                    }
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Executes the backward pass on a max pooling layer with a 2x2 window and a stride of 2
        /// </summary>
        /// <param name="x">The original input <see cref="Tensor"/> used during the forward pass</param>
        /// <param name="xInfo">The info on the input <see cref="Tensor"/></param>
        /// <param name="dy">The output error for the current layer</param>
        /// <param name="dx">The resulting backpropagated error</param>
        /// <exception cref="ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        public static unsafe void PoolingBackward(in Tensor x, in TensorInfo xInfo, in Tensor dy, in Tensor dx)
        {
            // Prepare the result tensor
            if (!dx.MatchShape(x)) throw new ArgumentException("The result tensor must have the same shape as the input", nameof(dx));
            int n = x.Entities, l = x.Length;
            if (n < 1 || l < 1) throw new ArgumentException("The input tensor isn't valid");
            int
                depth = xInfo.Channels,
                imgSize = l % depth == 0 ? l / depth : throw new ArgumentException("Invalid depth parameter for the input tensor", nameof(x)),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentException("The size of the input tensor isn't valid", nameof(x));
            int
                poolAxis = imgAxis / 2 + (imgAxis % 2 == 0 ? 0 : 1),
                poolSize = poolAxis * poolAxis,
                poolFinalWidth = depth * poolSize,
                edge = imgAxis - 1;
            int
                pn = dy.Entities,
                pl = dy.Length;
            if (pn != n || pl != poolFinalWidth) throw new ArgumentException("Invalid pooled tensor", nameof(dy));

            // Pooling kernel
            float* px = x, pdy = dy, pdx = dx;
            void Kernel(int sample)
            {
                int
                    sourceBaseOffset = sample * l,
                    resultBaseOffset = sample * poolFinalWidth;
                for (int z = 0; z < depth; z++)
                {
                    int
                        sourceZOffset = sourceBaseOffset + z * imgSize,
                        resultZOffset = resultBaseOffset + z * poolSize,
                        c = 0;
                    for (int i = 0; i < imgAxis; i += 2)
                    {
                        int
                            sourceIOffset = sourceZOffset + i * imgAxis,
                            resultXOffset = resultZOffset + c * poolAxis,
                            r = 0;
                        if (i == edge)
                        {
                            // Last row
                            for (int j = 0; j < imgAxis; j += 2)
                            {
                                if (j == l - 1)
                                {
                                    pdx[sourceIOffset + j] = pdy[resultXOffset + r++];
                                }
                                else
                                {
                                    float
                                        left = px[sourceIOffset + j],
                                        right = px[sourceIOffset + j + 1];
                                    if (left > right)
                                    {
                                        pdx[sourceIOffset + j] = pdy[resultXOffset + r++];
                                        pdx[sourceIOffset + j + 1] = 0;
                                    }
                                    else
                                    {
                                        pdx[sourceIOffset + j + 1] = pdy[resultXOffset + r++];
                                        pdx[sourceIOffset + j] = 0;
                                    }
                                }
                            }
                        }
                        else
                        {
                            int sourceI_1Offset = sourceZOffset + (i + 1) * imgAxis;
                            for (int j = 0; j < imgAxis; j += 2)
                            {
                                if (j == edge)
                                {
                                    // Last column
                                    float
                                        up = px[sourceIOffset + j],
                                        down = px[sourceI_1Offset + j];
                                    if (up > down)
                                    {
                                        pdx[sourceIOffset + j] = pdy[resultXOffset + r++];
                                        pdx[sourceI_1Offset + j] = 0;
                                    }
                                    else
                                    {
                                        pdx[sourceI_1Offset + j] = pdy[resultXOffset + r++];
                                        pdx[sourceIOffset + j] = 0;
                                    }
                                }
                                else
                                {
                                    int offset = sourceIOffset + j;
                                    float
                                        max = px[offset],
                                        next = px[sourceIOffset + j + 1];
                                    if (next > max)
                                    {
                                        max = next;
                                        pdx[offset] = 0;
                                        offset = sourceIOffset + j + 1;
                                    }
                                    else pdx[sourceIOffset + j + 1] = 0;
                                    next = px[sourceI_1Offset + j];
                                    if (next > max)
                                    {
                                        max = next;
                                        pdx[offset] = 0;
                                        offset = sourceI_1Offset + j;
                                    }
                                    else pdx[sourceI_1Offset + j] = 0;
                                    next = px[sourceI_1Offset + j + 1];
                                    if (next > max)
                                    {
                                        pdx[offset] = 0;
                                        offset = sourceI_1Offset + j + 1;
                                    }
                                    else pdx[sourceI_1Offset + j + 1] = 0;
                                    pdx[offset] = pdy[resultXOffset + r++];
                                }
                            }
                        }
                        c++;
                    }
                }
            }
            Parallel.For(0, n, Kernel).AssertCompleted();
        }
    }
}