using NeuralNetworkNET.Structs;
using System;
using System.Threading.Tasks;

namespace NeuralNetworkNET.Extensions
{
    /// <summary>
    /// A static class with a collection of pooling extension methods
    /// </summary>
    internal static class PoolingExtensions
    {
        /// <summary>
        /// Pools the input matrix with a window of 2 and a stride of 2
        /// </summary>
        /// <param name="source">The input matrix to pool</param>
        /// <param name="depth">The number of images for each matrix row</param>
        /// <param name="result">The resulting pooled matrix</param>
        public static unsafe void Pool2x2(in this Tensor source, int depth, out Tensor result)
        {
            // Prepare the result matrix
            if (depth < 1) throw new ArgumentOutOfRangeException(nameof(depth), "The number of images per sample must be at least equal to 1");
            int h = source.Entities, w = source.Length;
            if (h < 1 || w < 1) throw new ArgumentException("The input matrix isn't valid");
            int
                imgSize = w % depth == 0 ? w / depth : throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(source), "The size of the input matrix isn't valid");
            int
                poolAxis = imgAxis / 2 + (imgAxis % 2 == 0 ? 0 : 1),
                poolSize = poolAxis * poolAxis,
                poolFinalWidth = depth * poolSize,
                edge = imgAxis - 1;
            Tensor.New(h, poolFinalWidth, out result);

            // Pooling kernel
            float* psource = source, presult = result;
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
                        x = 0;
                    for (int i = 0; i < imgAxis; i += 2)
                    {
                        int
                            sourceIOffset = sourceZOffset + i * imgAxis,
                            resultXOffset = resultZOffset + x * poolAxis,
                            y = 0;
                        if (i == edge)
                        {
                            // Last row
                            for (int j = 0; j < imgAxis; j += 2)
                            {
                                float max;
                                if (j == w - 1) max = psource[sourceIOffset + j]; // Last column
                                else
                                {
                                    float
                                        left = psource[sourceIOffset + j],
                                        right = psource[sourceIOffset + j + 1];
                                    max = left > right ? left : right;
                                }
                                presult[resultXOffset + y++] = max;
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
                                        up = psource[sourceIOffset + j],
                                        down = psource[sourceI_1Offset + j];
                                    max = up > down ? up : down;
                                }
                                else
                                {
                                    float
                                        upLeft = psource[sourceIOffset + j],
                                        upRight = psource[sourceIOffset + j + 1],
                                        downLeft = psource[sourceI_1Offset + j],
                                        downRight = psource[sourceI_1Offset + j + 1],
                                        maxUp = upLeft > upRight ? upLeft : upRight,
                                        maxDown = downLeft > downRight ? downLeft : downRight;
                                    max = maxUp > maxDown ? maxUp : maxDown;
                                }
                                presult[resultXOffset + y++] = max;
                            }
                        }
                        x++;
                    }
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Pools the input matrix with a window of 2 and a stride of 2
        /// </summary>
        /// <param name="source">The activation matrix that will also hold the final result</param>
        /// <param name="pooled">The matrix to upscale according to the source values</param>
        /// <param name="depth">The number of images for each matrix row</param>
        public static unsafe void UpscalePool2x2(in this Tensor source, in Tensor pooled, int depth)
        {
            // Prepare the result matrix
            if (depth < 1) throw new ArgumentOutOfRangeException(nameof(depth), "The number of images per sample must be at least equal to 1");
            int h = source.Entities, w = source.Length;
            if (h < 1 || w < 1) throw new ArgumentException("The input matrix isn't valid");
            int
                imgSize = w % depth == 0 ? w / depth : throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(source), "The size of the input matrix isn't valid");
            int
                poolAxis = imgAxis / 2 + (imgAxis % 2 == 0 ? 0 : 1),
                poolSize = poolAxis * poolAxis,
                poolFinalWidth = depth * poolSize,
                edge = imgAxis - 1;
            int
                ph = pooled.Entities,
                pw = pooled.Length;
            if (ph != h || pw != poolFinalWidth) throw new ArgumentException("Invalid pooled matrix", nameof(pooled));

            // Pooling kernel
            float* psource = source, ppooled = pooled;
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
                        x = 0;
                    for (int i = 0; i < imgAxis; i += 2)
                    {
                        int
                            sourceIOffset = sourceZOffset + i * imgAxis,
                            resultXOffset = resultZOffset + x * poolAxis,
                            y = 0;
                        if (i == edge)
                        {
                            // Last row
                            for (int j = 0; j < imgAxis; j += 2)
                            {
                                if (j == w - 1)
                                {
                                    psource[sourceIOffset + j] = ppooled[resultXOffset + y++];
                                }
                                else
                                {
                                    float
                                        left = psource[sourceIOffset + j],
                                        right = psource[sourceIOffset + j + 1];
                                    if (left > right)
                                    {
                                        psource[sourceIOffset + j] = ppooled[resultXOffset + y++];
                                        psource[sourceIOffset + j + 1] = 0;
                                    }
                                    else
                                    {
                                        psource[sourceIOffset + j + 1] = ppooled[resultXOffset + y++];
                                        psource[sourceIOffset + j] = 0;
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
                                        up = psource[sourceIOffset + j],
                                        down = psource[sourceI_1Offset + j];
                                    if (up > down)
                                    {
                                        psource[sourceIOffset + j] = ppooled[resultXOffset + y++];
                                        psource[sourceI_1Offset + j] = 0;
                                    }
                                    else
                                    {
                                        psource[sourceI_1Offset + j] = ppooled[resultXOffset + y++];
                                        psource[sourceIOffset + j] = 0;
                                    }
                                }
                                else
                                {
                                    int offset = sourceIOffset + j;
                                    float
                                        max = psource[offset],
                                        next = psource[sourceIOffset + j + 1];
                                    if (next > max)
                                    {
                                        max = next;
                                        psource[offset] = 0;
                                        offset = sourceIOffset + j + 1;
                                    }
                                    else psource[sourceIOffset + j + 1] = 0;
                                    next = psource[sourceI_1Offset + j];
                                    if (next > max)
                                    {
                                        max = next;
                                        psource[offset] = 0;
                                        offset = sourceI_1Offset + j;
                                    }
                                    else psource[sourceI_1Offset + j] = 0;
                                    next = psource[sourceI_1Offset + j + 1];
                                    if (next > max)
                                    {
                                        psource[offset] = 0;
                                        offset = sourceI_1Offset + j + 1;
                                    }
                                    else psource[sourceI_1Offset + j + 1] = 0;
                                    psource[offset] = ppooled[resultXOffset + y++];
                                }
                            }
                        }
                        x++;
                    }
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
        }
    }
}