using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkDotNet.Core.APIs.Models;
using NeuralNetworkDotNet.Core.Helpers;

namespace NeuralNetworkDotNet.Cpu.cpuDNN
{
    public static partial class CpuDnn
    {
        /// <summary>
        /// Executes the forward pass on a max pooling layer with a 2x2 window and a stride of 2
        /// </summary>
        /// <param name="x">The input <see cref="Tensor"/> to pool</param>
        /// <param name="y">The resulting pooled <see cref="Tensor"/></param>
        /// <exception cref="ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        public static void PoolingForward([NotNull] Tensor x, [NotNull] Tensor y)
        {
            Guard.IsFalse(x.NCHW < 1, nameof(x), "The input tensor isn't valid");
            Guard.IsTrue(x.H == x.W, nameof(x), "The input tensor must contain square images");

            int
                h = x.N,
                w = x.CHW,
                depth = x.C,
                imgSize = x.HW,
                imgAxis = x.H,  // Size of an edge of one of the inner images per sample
                poolAxis = imgAxis / 2 + (imgAxis % 2 == 0 ? 0 : 1),
                poolSize = poolAxis * poolAxis,
                poolFinalWidth = depth * poolSize,
                edge = imgAxis - 1;

            Guard.IsTrue(y.Shape == (x.N, x.C, poolAxis, poolAxis), nameof(y), "The output tensor doesn't have the right shape");

            // Pooling kernel
            void Kernel(int sample)
            {
                int
                    sourceBaseOffset = sample * w,
                    resultBaseOffset = sample * poolFinalWidth;

                ref var rx = ref x.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                for (var z = 0; z < depth; z++)
                {
                    int
                        sourceZOffset = sourceBaseOffset + z * imgSize,
                        resultZOffset = resultBaseOffset + z * poolSize,
                        c = 0;
                    for (var i = 0; i < imgAxis; i += 2)
                    {
                        int
                            sourceIOffset = sourceZOffset + i * imgAxis,
                            resultXOffset = resultZOffset + c * poolAxis,
                            r = 0;
                        if (i == edge)
                        {
                            // Last row
                            for (var j = 0; j < imgAxis; j += 2)
                            {
                                float max;
                                if (j == w - 1) max = Unsafe.Add(ref rx, sourceIOffset + j); // Last column
                                else
                                {
                                    float
                                        left = Unsafe.Add(ref rx, sourceIOffset + j),
                                        right = Unsafe.Add(ref rx, sourceIOffset + j + 1);
                                    max = left > right ? left : right;
                                }

                                Unsafe.Add(ref ry, resultXOffset + r++) = max;
                            }
                        }
                        else
                        {
                            var sourceI_1Offset = sourceZOffset + (i + 1) * imgAxis;
                            for (var j = 0; j < imgAxis; j += 2)
                            {
                                float max;
                                if (j == edge)
                                {
                                    // Last column
                                    float
                                        up = Unsafe.Add(ref rx, sourceIOffset + j),
                                        down = Unsafe.Add(ref rx, sourceI_1Offset + j);
                                    max = up > down ? up : down;
                                }
                                else
                                {
                                    float
                                        upLeft = Unsafe.Add(ref rx, sourceIOffset + j),
                                        upRight = Unsafe.Add(ref rx, sourceIOffset + j + 1),
                                        downLeft = Unsafe.Add(ref rx, sourceI_1Offset + j),
                                        downRight = Unsafe.Add(ref rx, sourceI_1Offset + j + 1),
                                        maxUp = upLeft > upRight ? upLeft : upRight,
                                        maxDown = downLeft > downRight ? downLeft : downRight;
                                    max = maxUp > maxDown ? maxUp : maxDown;
                                }

                                Unsafe.Add(ref ry, resultXOffset + r++) = max;
                            }
                        }

                        c++;
                    }
                }
            }

            Parallel.For(0, h, Kernel);
        }

        /// <summary>
        /// Executes the backward pass on a max pooling layer with a 2x2 window and a stride of 2
        /// </summary>
        /// <param name="x">The original input <see cref="Tensor"/> used during the forward pass</param>
        /// <param name="dy">The output error for the current layer</param>
        /// <param name="dx">The resulting backpropagated error</param>
        /// <exception cref="ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        public static void PoolingBackward([NotNull] Tensor x, [NotNull] Tensor dy, [NotNull] Tensor dx)
        {
            Guard.IsFalse(x.NCHW < 1, nameof(x), "The input tensor isn't valid");
            Guard.IsTrue(dx.Shape == x.Shape, nameof(dx), "The result tensor must have the same shape as the input");
            Guard.IsTrue(x.H == x.W, nameof(x), "The input tensor must contain square images");

            int
                n = x.N,
                l = x.CHW,
                depth = x.C,
                imgSize = x.HW,
                imgAxis = x.H,  // Size of an edge of one of the inner images per sample
                poolAxis = imgAxis / 2 + (imgAxis % 2 == 0 ? 0 : 1),
                poolSize = poolAxis * poolAxis,
                poolFinalWidth = depth * poolSize,
                edge = imgAxis - 1,
                pn = dy.N,
                pl = dy.CHW;

            Guard.IsFalse(pn != n || pl != poolFinalWidth, nameof(dy), "Invalid pooled tensor shape");

            // Pooling kernel
            void Kernel(int sample)
            {
                int
                    sourceBaseOffset = sample * l,
                    resultBaseOffset = sample * poolFinalWidth;

                ref var rx = ref x.Span.GetPinnableReference();
                ref var rdy = ref dy.Span.GetPinnableReference();
                ref var rdx = ref dx.Span.GetPinnableReference();

                for (var z = 0; z < depth; z++)
                {
                    int
                        sourceZOffset = sourceBaseOffset + z * imgSize,
                        resultZOffset = resultBaseOffset + z * poolSize,
                        c = 0;
                    for (var i = 0; i < imgAxis; i += 2)
                    {
                        int
                            sourceIOffset = sourceZOffset + i * imgAxis,
                            resultXOffset = resultZOffset + c * poolAxis,
                            r = 0;
                        if (i == edge)
                        {
                            // Last row
                            for (var j = 0; j < imgAxis; j += 2)
                            {
                                if (j == l - 1)
                                {
                                    Unsafe.Add(ref rdx, sourceIOffset + j) = Unsafe.Add(ref rdy, resultXOffset + r++);
                                }
                                else
                                {
                                    float
                                        left = Unsafe.Add(ref rx, sourceIOffset + j),
                                        right = Unsafe.Add(ref rx, sourceIOffset + j + 1);
                                    if (left > right)
                                    {
                                        Unsafe.Add(ref rdx, sourceIOffset + j) = Unsafe.Add(ref rdy, resultXOffset + r++);
                                        Unsafe.Add(ref rdx, sourceIOffset + j + 1) = 0;
                                    }
                                    else
                                    {
                                        Unsafe.Add(ref rdx, sourceIOffset + j + 1) = Unsafe.Add(ref rdy, resultXOffset + r++);
                                        Unsafe.Add(ref rdx, sourceIOffset + j) = 0;
                                    }
                                }
                            }
                        }
                        else
                        {
                            var sourceI_1Offset = sourceZOffset + (i + 1) * imgAxis;
                            for (var j = 0; j < imgAxis; j += 2)
                            {
                                if (j == edge)
                                {
                                    // Last column
                                    float
                                        up = Unsafe.Add(ref rx, sourceIOffset + j),
                                        down = Unsafe.Add(ref rx, sourceI_1Offset + j);
                                    if (up > down)
                                    {
                                        Unsafe.Add(ref rdx, sourceIOffset + j) = Unsafe.Add(ref rdy, resultXOffset + r++);
                                        Unsafe.Add(ref rdx, sourceI_1Offset + j) = 0;
                                    }
                                    else
                                    {
                                        Unsafe.Add(ref rdx, sourceI_1Offset + j) = Unsafe.Add(ref rdy, resultXOffset + r++);
                                        Unsafe.Add(ref rdx, sourceIOffset + j) = 0;
                                    }
                                }
                                else
                                {
                                    var offset = sourceIOffset + j;
                                    float
                                        max = Unsafe.Add(ref rx, offset),
                                        next = Unsafe.Add(ref rx, sourceIOffset + j + 1);
                                    if (next > max)
                                    {
                                        max = next;
                                        Unsafe.Add(ref rdx, offset) = 0;
                                        offset = sourceIOffset + j + 1;
                                    }
                                    else Unsafe.Add(ref rdx, sourceIOffset + j + 1) = 0;
                                    next = Unsafe.Add(ref rx, sourceI_1Offset + j);
                                    if (next > max)
                                    {
                                        max = next;
                                        Unsafe.Add(ref rdx, offset) = 0;
                                        offset = sourceI_1Offset + j;
                                    }
                                    else Unsafe.Add(ref rdx, sourceI_1Offset + j) = 0;
                                    next = Unsafe.Add(ref rx, sourceI_1Offset + j + 1);
                                    if (next > max)
                                    {
                                        Unsafe.Add(ref rdx, offset) = 0;
                                        offset = sourceI_1Offset + j + 1;
                                    }
                                    else Unsafe.Add(ref rdx, sourceI_1Offset + j + 1) = 0;
                                    Unsafe.Add(ref rdx, offset) = Unsafe.Add(ref rdy, resultXOffset + r++);
                                }
                            }
                        }

                        c++;
                    }
                }
            }

            Parallel.For(0, n, Kernel);
        }
    }
}
