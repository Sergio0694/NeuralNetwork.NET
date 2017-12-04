using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Structs;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// Test class for the <see cref="ConvolutionExtensions"/> class
    /// </summary>
    [TestClass]
    [TestCategory(nameof(ConvolutionExtensions))]
    public class ConvolutionExtensionsTest
    {
        [TestMethod]
        public unsafe void Pool1()
        {
            // Down
            float[,]
                m =
                {
                    {
                        -1, 0, 1, 2,
                        1, 1, 1, 1,
                        0, -0.3f, -5, -0.5f,
                        -1, 10, -2, -1
                    }
                },
                r =
                {
                    {
                        1, 2,
                        10, -0.5f
                    }
                };
            fixed (float* pm = m)
            {
                FloatSpan2D.Fix(pm, 1, 16, out FloatSpan2D mSpan);
                mSpan.Pool2x2(1, out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(r));

                // Upscale
                mSpan.UpscalePool2x2(result, 1);
                float[,] expected =
                {
                    {
                        0, 0, 0, 2,
                        1, 0, 0, 0,
                        0, 0, 0, -0.5f,
                        0, 10, 0, 0
                    }
                };
                Assert.IsTrue(mSpan.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void Pool2()
        {
            // Test values
            float[,]
                m =
                {
                    {
                        0.77f, -0.11f, 0.11f, 0.33f, 0.55f, -0.11f, 0.33f,
                        -0.11f, 1, -0.11f, 0.33f, -0.11f, 0.11f, -0.11f,
                        0.11f, -0.11f, 1, -0.33f, 0.11f, -0.11f, 0.55f,
                        0.33f, 0.33f, -0.33f, 0.55f, -0.33f, 0.33f, 0.33f,
                        0.55f, -0.11f, 0.11f, -0.33f, 1, -0.11f, 0.11f,
                        -0.11f, 0.11f, -0.11f, 0.33f, -0.11f, 1, -0.11f,
                        0.33f, -0.11f, 0.55f, 0.33f, 0.11f, -0.11f, 0.77f
                    }
                },
                r =
                {
                    {
                        1, 0.33f, 0.55f, 0.33f,
                        0.33f, 1, 0.33f, 0.55f,
                        0.55f, 0.33f, 1, 0.11f,
                        0.33f, 0.55f, 0.11f, 0.77f
                    }
                };
            fixed (float* pm = m)
            {
                FloatSpan2D.Fix(pm, 1, 49, out FloatSpan2D mSpan);
                mSpan.Pool2x2(1, out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(r));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void Pool3()
        {
            // Test values
            float[,]
                m =
                {
                    {
                        -1, 0,
                        1, 1
                    },
                },
                r =
                {
                    { 1 }
                };
            fixed (float* pm = m)
            {
                FloatSpan2D.Fix(pm, 1, 4, out FloatSpan2D mSpan);
                mSpan.Pool2x2(1, out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(r));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void Pool4()
        {
            // Test values
            float[,]
                m =
                {
                    {
                        -1, 0, 1, 2,
                        1, 1, 1, 1, 0,
                        -0.3f, -5, -0.5f,
                        -1, 10, -2, -1
                    },
                    {
                        -1, 0, 1, 2,
                        1, 1, 1, 1, 0,
                        -0.3f, -5, 1.2f,
                        -1, 10, -2, -1
                    }
                },
                r =
                {
                    {
                        1, 2,
                        10, -0.5f
                    },
                    {
                        1, 2,
                        10, 1.2f
                    },
                };
            fixed (float* pm = m)
            {
                FloatSpan2D.Fix(pm, 2, 16, out FloatSpan2D mSpan);
                mSpan.Pool2x2(1, out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(r));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void Pool5()
        {
            // Test values
            float[,]
                m =
                {
                    {
                        -1, 0, 1, 2,
                        1, 1, 1, 1,
                        0, -0.3f, -5, -0.5f,
                        -1, 10, -2, -1,

                        -1, 0, 1, 2,
                        1, 1, 1, 1,
                        0, -0.3f, -5, 1.2f,
                        -1, 10, -2, -1
                    },
                    {
                        -1, 0, 1, 2,
                        1, 1, 1, 1,
                        0, -0.3f, -5, 1.2f,
                        -1, 10, -2, -1,

                        -1, 0, 1, 2,
                        1, 1, 1, 1,
                        0, -0.3f, -5, 1.45f,
                        -1, 10, -2, -1
                    }
                },
                r =
                {
                    {
                        1, 2,
                        10, -0.5f,

                        1, 2,
                        10, 1.2f
                    },
                    {
                        1, 2,
                        10, 1.2f,

                        1, 2,
                        10, 1.45f
                    },
                };
            fixed (float* pm = m)
            {
                FloatSpan2D.Fix(pm, 2, 32, out FloatSpan2D mSpan);
                mSpan.Pool2x2(2, out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(r));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void Rotate1()
        {
            // Test values
            float[,]
                m =
                {
                    {
                        1, 2,
                        3, 4
                    }
                },
                r =
                {
                    {
                        4, 3,
                        2, 1
                    }
                };
            fixed (float* pm = m)
            {
                FloatSpan2D.Fix(pm, 1, 4, out FloatSpan2D span);
                span.Rotate180(1, out FloatSpan2D tSpan);
                Assert.IsTrue(tSpan.ToArray2D().ContentEquals(r));
                tSpan.Free();
            }
        }
        
        [TestMethod]
        public unsafe void Rotate2()
        {
            // Test values
            float[,]
                m =
                {
                    {
                        1, 2, 3,
                        4, 5, 6,
                        7, 8, 9
                    }
                },
                r =
                {
                    {
                        9, 8, 7,
                        6, 5, 4,
                        3, 2, 1
                    }
                };
            fixed (float* pm = m)
            {
                FloatSpan2D.Fix(pm, 1, 9, out FloatSpan2D span);
                span.Rotate180(1, out FloatSpan2D tSpan);
                Assert.IsTrue(tSpan.ToArray2D().ContentEquals(r));
                tSpan.Free();
            }
        }

        [TestMethod]
        public unsafe void Rotate3()
        {
            // Test values
            float[,]
                m =
                {
                    {
                        1, 2, 3,
                        4, 5, 6,
                        7, 8, 9,

                        1, 99, 3,
                        4, 5, 6,
                        7, 8, 9
                    },
                    {
                        1, 2, 3,
                        4, 5, 66,
                        7, 8, 9,

                        1, 2, 3,
                        44, 5, 6,
                        7, 8, 9
                    }
                },
                r =
                {
                    {
                        9, 8, 7,
                        6, 5, 4,
                        3, 2, 1,

                        9, 8, 7,
                        6, 5, 4,
                        3, 99, 1
                    },
                    {
                        9, 8, 7,
                        66, 5, 4,
                        3, 2, 1,

                        9, 8, 7,
                        6, 5, 44,
                        3, 2, 1
                    },
                };
            fixed (float* pm = m)
            {
                FloatSpan2D.Fix(pm, 2, 18, out FloatSpan2D span);
                span.Rotate180(2, out FloatSpan2D tSpan);
                Assert.IsTrue(tSpan.ToArray2D().ContentEquals(r));
                tSpan.Free();
            }
        }

        [TestMethod]
        public unsafe void UpscalePool1()
        {
            float[,]
                m =
                {
                    {
                        -1, 0, 1, 2,
                        1.2f, 1, 1, 1,
                        0, -0.3f, -5, -0.5f,
                        -1, 10, -2, -1
                    }
                },
                p =
                {
                    {
                        66, 77,
                        99, 11
                    }
                },
                r =
                {
                    {
                        0, 0, 0, 77,
                        66, 0, 0, 0,
                        0, 0, 0, 11,
                        0, 99, 0, 0
                    }
                };
            fixed (float* pm = m, pp = p)
            {
                FloatSpan2D.Fix(pm, 1, 16, out FloatSpan2D mSpan);
                FloatSpan2D.Fix(pp, 1, 4, out FloatSpan2D pSpan);
                mSpan.UpscalePool2x2(pSpan, 1);
                Assert.IsTrue(mSpan.ToArray2D().ContentEquals(r));
            }
        }

        [TestMethod]
        public unsafe void UpscalePool2()
        {
            float[,]
                m =
                {
                    {
                        -1, 0, 1, 2,
                        1.2f, 1, 1, 1,
                        0, -0.3f, -5, -0.5f,
                        -1, 10, -2, -1,

                        -1, 0, 1, 2,
                        1.2f, 1, 1, 1,
                        0, -0.3f, -5, -0.5f,
                        11, 10, -2, -1
                    }
                },
                p =
                {
                    {
                        66, 77,
                        99, 11,

                        66, 1,
                        111, 11
                    }
                },
                r =
                {
                    {
                        0, 0, 0, 77,
                        66, 0, 0, 0,
                        0, 0, 0, 11,
                        0, 99, 0, 0,

                        0, 0, 0, 1,
                        66, 0, 0, 0,
                        0, 0, 0, 11,
                        111, 0, 0, 0
                    }
                };
            fixed (float* pm = m, pp = p)
            {
                FloatSpan2D.Fix(pm, 1, 32, out FloatSpan2D mSpan);
                FloatSpan2D.Fix(pp, 1, 8, out FloatSpan2D pSpan);
                mSpan.UpscalePool2x2(pSpan, 2);
                Assert.IsTrue(mSpan.ToArray2D().ContentEquals(r));
            }
        }

        [TestMethod]
        public unsafe void UpscalePool3()
        {
            float[,]
                m =
                {
                    {
                        -1, 0, 1, 2,
                        1.2f, 1, 1, 1,
                        0, -0.3f, -5, -0.5f,
                        -1, 10, -2, -1,

                        -1, 0, 1, 2,
                        1.2f, 1, 1, 1,
                        0, -0.3f, -5, -0.5f,
                        11, 10, -2, -1
                    },
                    {
                        -1, 0, 1, 2,
                        1.2f, 1, 1, 1,
                        0, -0.3f, -5, -0.5f,
                        -1, 10, -2, -1,

                        -1, 0, 1, 2,
                        1.2f, 1, 1, 1,
                        0, 144, -5, -0.5f,
                        11, 10, -2, -1
                    }
                },
                p =
                {
                    {
                        66, 77,
                        99, 11,

                        66, 1,
                        111, 11
                    },
                    {
                        66, 77,
                        99, 11,

                        66, 1,
                        111, 11
                    }
                },
                r =
                {
                    {
                        0, 0, 0, 77,
                        66, 0, 0, 0,
                        0, 0, 0, 11,
                        0, 99, 0, 0,

                        0, 0, 0, 1,
                        66, 0, 0, 0,
                        0, 0, 0, 11,
                        111, 0, 0, 0
                    },
                    {
                        0, 0, 0, 77,
                        66, 0, 0, 0,
                        0, 0, 0, 11,
                        0, 99, 0, 0,

                        0, 0, 0, 1,
                        66, 0, 0, 0,
                        0, 111, 0, 11,
                        0, 0, 0, 0
                    }
                };
            fixed (float* pm = m, pp = p)
            {
                FloatSpan2D.Fix(pm, 2, 32, out FloatSpan2D mSpan);
                FloatSpan2D.Fix(pp, 2, 8, out FloatSpan2D pSpan);
                mSpan.UpscalePool2x2(pSpan, 2);
                Assert.IsTrue(mSpan.ToArray2D().ContentEquals(r));
            }
        }

        [TestMethod]
        public unsafe void UpscalePool4()
        {
            float[,]
                m =
                {
                    {
                        -1, 0, 1, 2,
                        1.2f, 1, 1, 1,
                        0, -0.3f, -5, -0.5f,
                        -1, 10, -2, -1,

                        -1, 2, 1, 2,
                        1.2f, 1, 1, 1,
                        0, -0.3f, 0, -0.5f,
                        11, 10, -2, -1,

                        -1, 2, 1, 2,
                        1.2f, 5, 1, 5,
                        0, 22, 0, -0.5f,
                        11, 10, -2, 7
                    },
                    {
                        -1, 0, 1, 2,
                        1.2f, 1, 1, 1,
                        0, -0.3f, -5, -0.5f,
                        -1, 10, -2, -1,

                        -1, 2, 1, 2,
                        1.2f, 1, 1, 1,
                        0, -0.3f, 0, -0.5f,
                        11, 10, -2, -1,

                        99, 2, 1, 2,
                        1.2f, 5, 1, 5,
                        0, 22, 0, -0.5f,
                        11, 10, -2, 7
                    }
                },
                p =
                {
                    {
                        66, 77,
                        99, 11,

                        66, 1,
                        111, 11,

                        11, 22,
                        33, 44
                    },
                    {
                        66, 77,
                        222, 11,

                        66, 1,
                        111, 11,

                        11, 22,
                        33, 44
                    }
                },
                r =
                {
                    {
                        0, 0, 0, 77,
                        66, 0, 0, 0,
                        0, 0, 0, 11,
                        0, 99, 0, 0,

                        0, 66, 0, 1,
                        0, 0, 0, 0,
                        0, 0, 11, 0,
                        111, 0, 0, 0,

                        0, 0, 0, 0,
                        0, 11, 0, 22,
                        0, 33, 0, 0,
                        0, 0, 0, 44
                    },
                    {
                        0, 0, 0, 77,
                        66, 0, 0, 0,
                        0, 0, 0, 11,
                        0, 222, 0, 0,

                        0, 66, 0, 1,
                        0, 0, 0, 0,
                        0, 0, 11, 0,
                        111, 0, 0, 0,

                        11, 0, 0, 0,
                        0, 0, 0, 22,
                        0, 33, 0, 0,
                        0, 0, 0, 44
                    }
                };
            fixed (float* pm = m, pp = p)
            {
                FloatSpan2D.Fix(pm, 2, 48, out FloatSpan2D mSpan);
                FloatSpan2D.Fix(pp, 2, 12, out FloatSpan2D pSpan);
                mSpan.UpscalePool2x2(pSpan, 3);
                Assert.IsTrue(mSpan.ToArray2D().ContentEquals(r));
            }
        }

        [TestMethod]
        public unsafe void Compress1()
        {
            // Test values
            float[,]
                m =
                {
                    {
                        1, 2, 3,
                        4, 5, 6,
                        7, 8, 9
                    }
                };
            float[] r = { 45 };
            fixed (float* pm = m)
            {
                FloatSpan2D.Fix(pm, 1, 9, out FloatSpan2D mSpan);
                mSpan.CompressVertically(1, out FloatSpan v);
                Assert.IsTrue(v.ToArray().ContentEquals(r));
                v.Free();
            }
        }

        [TestMethod]
        public unsafe void Compress2()
        {
            // Test values
            float[,]
                m =
                {
                    {
                        1, 2, 3,
                        4, 5, 6,
                        7, 8, 9,

                        1, 99, 3,
                        4, 5, 6,
                        7, 8, 9
                    },
                    {
                        1, 2, 3,
                        4, 5, 66,
                        7, 8, 9,

                        1, 2, 3,
                        44, 5, 6,
                        7, 8, 9
                    }
                };
            float[] r = { 150, 227 };
            fixed (float* pm = m)
            {
                FloatSpan2D.Fix(pm, 2, 18, out FloatSpan2D mSpan);
                mSpan.CompressVertically(2, out FloatSpan v);
                Assert.IsTrue(v.ToArray().ContentEquals(r));
                v.Free();
            }
        }

        // 1-depth, 3*3 with 2*2 = 2*2 result
        [TestMethod]
        public unsafe void Convolution2DValid1()
        {
            float[,]
                l =
                {
                    {
                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0
                    }
                },
                k =
                {
                    {
                        1, 1,
                        0, 1
                    }
                };
            float[] b = { 0.6f };
            float[,] expected =
            {
                {
                    2.6f, 2.6f,
                    4.6f, 1.6f
                }
            };
            fixed (float* pl = l)
            {
                FloatSpan2D.Fix(pl, 1, 9, out FloatSpan2D lSpan);
                lSpan.ConvoluteForward((3, 3, 1), k, (2, 2, 1), b, out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }

        // 1-depth, 2 sample 3*3 with 2*2 = 2 sample 2*2 result
        [TestMethod]
        public unsafe void Convolution2DValid2()
        {
            float[,]
                l =
                {
                    {
                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0
                    },
                    {
                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0
                    }
                },
                k =
                {
                    {
                        1, 1,
                        0, 1
                    }
                };
            float[,] expected =
            {
                {
                    2, 2,
                    4, 1
                },
                {
                    2, 2,
                    4, 1
                }
            };
            fixed (float* pl = l)
            {
                FloatSpan2D.Fix(pl, 2, 9, out FloatSpan2D lSpan);
                lSpan.ConvoluteForward((3, 3, 1), k, (2, 2, 1), new float[1], out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }

        // 1-depth, 3*3 with 2 kernels 2*2 = 2-depth 2*2 result
        [TestMethod]
        public unsafe void Convolution2DValid3()
        {
            float[,]
                l =
                {
                    {
                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0
                    }
                },
                k =
                {
                    {
                        1, 1,
                        0, 1
                    },
                    {
                        0, 1,
                        2, 0
                    }
                };
            float[,] expected =
            {
                {
                    3, 3,
                    5, 2,

                    4.5f, 0.5f,
                    1.5f, 3.5f
                }
            };
            fixed (float* pl = l)
            {
                FloatSpan2D.Fix(pl, 1, 9, out FloatSpan2D lSpan);
                lSpan.ConvoluteForward((3, 3, 1), k, (2, 2, 1), new[] { 1, 0.5f }, out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }

        // 2-depth, 3*3 with 2-depth kernel = 2*2 result
        [TestMethod]
        public unsafe void Convolution2DValid4()
        {
            float[,]
                l =
                {
                    {
                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0,

                        1, 0, 0,
                        0, 2, 1,
                        0, 1, 1
                    }
                },
                k =
                {
                    {
                        1, 1,
                        0, 1,

                        0, 1,
                        1, 0
                    }
                };
            float[,] expected =
            {
                {
                    2.1f, 4.1f,
                    6.1f, 3.1f
                }
            };
            fixed (float* pl = l)
            {
                FloatSpan2D.Fix(pl, 1, 18, out FloatSpan2D lSpan);
                lSpan.ConvoluteForward((3, 3, 2), k, (2, 2, 2), new[] { 0.1f }, out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void Convolution2DValid5()
        {
            float[,]
                l =
                {
                    {
                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0,

                        1, 0, 0,
                        0, 2, 1,
                        0, 1, 1
                    }
                },
                k =
                {
                    {
                        1, 1,
                        0, 1,

                        0, 1,
                        1, 0
                    },
                    {
                        1, 1,
                        0, 1,

                        0, 1,
                        1, 0
                    }
                };
            float[,] expected =
            {
                {
                    2, 4,
                    6, 3,

                    2.2f, 4.2f,
                    6.2f, 3.2f
                }
            };
            fixed (float* pl = l)
            {
                FloatSpan2D.Fix(pl, 1, 18, out FloatSpan2D lSpan);
                lSpan.ConvoluteForward((3, 3, 2), k, (2, 2, 2), new[] { 0, 0.2f }, out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void Convolution2DValidRectangle1()
        {
            float[,]
                l =
                {
                    {
                        0, 1, 0,
                        2, 0, 1
                    }
                },
                k =
                {
                    {
                        1, 1,
                        0, 1
                    }
                };
            float[,] expected =
            {
                {
                    2.9f, 2.9f
                }
            };
            fixed (float* pl = l)
            {
                FloatSpan2D.Fix(pl, 1, 6, out FloatSpan2D lSpan);
                lSpan.ConvoluteForward((2, 3, 1), k, (2, 2, 1), new[] { 0.9f }, out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void ConvolutionFull1()
        {
            float[,]
                l =
                {
                    {
                        0, 1,
                        -1, 2
                    }
                },
                k =
                {
                    {
                        1, 1,
                        0, 1
                    }
                };
            float[,] expected =
            {
                {
                    0, 1, 1,
                    -1, 1, 3,
                    0, -1, 2
                }
            };
            fixed (float* pl = l, pk = k)
            {
                FloatSpan2D.Fix(pl, 1, 4, out FloatSpan2D lSpan);
                FloatSpan2D.Fix(pk, 1, 4, out FloatSpan2D kSpan);
                lSpan.ConvoluteBackwards((2, 2, 1), kSpan, (2, 2, 1), out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void ConvolutionFull2()
        {
            float[,]
                l =
                {
                    {
                        0, 1,
                        -1, 2,

                        0, -1,
                        -1, 2
                    }
                },
                k =
                {
                    {
                        1, 1,
                        0, 1
                    },
                    {
                        1, 1,
                        0, 1
                    }
                };
            float[,] expected =
            {
                {
                    0, 0, 0,
                    -2, 2, 4,
                    0, -2, 4
                }
            };
            fixed (float* pl = l, pk = k)
            {
                FloatSpan2D.Fix(pl, 1, 8, out FloatSpan2D lSpan);
                FloatSpan2D.Fix(pk, 2, 4, out FloatSpan2D kSpan);
                lSpan.ConvoluteBackwards((2, 2, 2), kSpan, (2, 2, 1), out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void ConvolutionFull3()
        {
            float[,]
                l =
                {
                    {
                        0, 1,
                        -1, 2
                    }
                },
                k =
                {
                    {
                        1, 1,
                        0, 1,

                        1, 0,
                        1, 0
                    }
                };
            float[,] expected =
            {
                {
                    0, 1, 1,
                    -1, 1, 3,
                    0, -1, 2,

                    0, 1, 0,
                    -1, 3, 0,
                    -1, 2, 0
                }
            };
            fixed (float* pl = l, pk = k)
            {
                FloatSpan2D.Fix(pl, 1, 4, out FloatSpan2D lSpan);
                FloatSpan2D.Fix(pk, 1, 8, out FloatSpan2D kSpan);
                lSpan.ConvoluteBackwards((2, 2, 1), kSpan, (2, 2, 2), out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void ConvolutionFull4()
        {
            float[,]
                l =
                {
                    {
                        0, 1,
                        -1, 2,

                        0, -1,
                        -1, 2
                    },
                    {
                        0, 1,
                        2, 3,

                        -1, -1,
                        0, 4
                    }
                },
                k =
                {
                    {
                        1, 1,
                        0, 1,

                        1, 0,
                        1, 0
                    },
                    {
                        1, 0,
                        0, 1,

                        -2, 0,
                        3, 1
                    }
                };
            float[,] expected =
            {
                {
                    0, 0, 1,
                    -2, 3, 2,
                    0, -2, 4,

                    0, 3, 0,
                    1, -4, -1,
                    -4, 7, 2
                },
                {
                    -1, 0, 1,
                    2, 8, 3,
                    0, 2, 7,

                    2, 3, 0,
                    -1, -8, -1,
                    2, 15, 4
                }
            };
            fixed (float* pl = l, pk = k)
            {
                FloatSpan2D.Fix(pl, 2, 8, out FloatSpan2D lSpan);
                FloatSpan2D.Fix(pk, 2, 8, out FloatSpan2D kSpan);
                lSpan.ConvoluteBackwards((2, 2, 2), kSpan, (2, 2, 2), out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void Convolution2DGradient1()
        {
            float[,]
                l =
                {
                    {
                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0
                    }
                },
                k =
                {
                    {
                        1, 1,
                        0, 1
                    }
                };
            float[,] expected =
            {
                {
                    2, 2,
                    4, 1
                }
            };
            fixed (float* pl = l, pk = k)
            {
                FloatSpan2D.Fix(pl, 1, 9, out FloatSpan2D lSpan);
                FloatSpan2D.Fix(pk, 1, 4, out FloatSpan2D kSpan);
                lSpan.ConvoluteGradient((3, 3, 1), kSpan, (2, 2, 1), out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void Convolution2DGradient2()
        {
            float[,]
                l =
                {
                    {
                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0
                    }
                },
                k =
                {
                    {
                        1, 1,
                        0, 1,

                        1, 2,
                        0, 1
                    }
                };
            float[,] expected =
            {
                {
                    2, 2,
                    4, 1,

                    4, 2,
                    5, 2
                }
            };
            fixed (float* pl = l, pk = k)
            {
                FloatSpan2D.Fix(pl, 1, 9, out FloatSpan2D lSpan);
                FloatSpan2D.Fix(pk, 1, 8, out FloatSpan2D kSpan);
                lSpan.ConvoluteGradient((3, 3, 1), kSpan, (2, 2, 2), out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void Convolution2DGradient3()
        {
            float[,]
                l =
                {
                    {
                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0,

                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0
                    }
                },
                k =
                {
                    {
                        1, 1,
                        0, 1
                    }
                };
            float[,] expected =
            {
                {
                    2, 2,
                    4, 1,

                    2, 2,
                    4, 1
                }
            };
            fixed (float* pl = l, pk = k)
            {
                FloatSpan2D.Fix(pl, 1, 18, out FloatSpan2D lSpan);
                FloatSpan2D.Fix(pk, 1, 4, out FloatSpan2D kSpan);
                lSpan.ConvoluteGradient((3, 3, 2), kSpan, (2, 2, 1), out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void Convolution2DGradient4()
        {
            float[,]
                l =
                {
                    {
                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0,

                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0
                    },
                    {
                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0,

                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0
                    }
                },
                k =
                {
                    {
                        1, 1,
                        0, 1
                    },
                    {
                        1, 2,
                        0, 1
                    }
                };
            float[,] expected =
            {
                {
                    2, 2,
                    4, 1,

                    2, 2,
                    4, 1
                },
                {
                    4, 2,
                    5, 2,

                    4, 2,
                    5, 2
                }
            };
            fixed (float* pl = l, pk = k)
            {
                FloatSpan2D.Fix(pl, 2, 18, out FloatSpan2D lSpan);
                FloatSpan2D.Fix(pk, 2, 4, out FloatSpan2D kSpan);
                lSpan.ConvoluteGradient((3, 3, 2), kSpan, (2, 2, 1), out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }

        [TestMethod]
        public unsafe void Convolution2DGradient5()
        {
            float[,]
                l =
                {
                    {
                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0,

                        1, 0, 1,
                        0, 2, 1,
                        0, 1, 1
                    },
                    {
                        0, -1, 0,
                        2, 0, 1,
                        1, 1, 0,

                        1, 0, 1,
                        0, 2, 1,
                        0, 1, 1
                    }
                },
                k =
                {
                    {
                        1, 1,
                        0, 1,

                        0, 1,
                        1, 0
                    },
                    {
                        1, 1,
                        0, 1,

                        0, -1,
                        1, 0
                    }
                };
            float[,] expected =
            {
                {
                    2, 2,
                    4, 1,

                    3, 3,
                    1, 4,

                    3, 0,
                    1, 2,

                    0, 3,
                    2, 2
                },
                {
                    2, 0,
                    4, 1,

                    3, 3,
                    1, 4,

                    -3, 0,
                    -1, 0,

                    0, -1,
                    2, 0
                }
            };
            fixed (float* pl = l, pk = k)
            {
                FloatSpan2D.Fix(pl, 2, 18, out FloatSpan2D lSpan);
                FloatSpan2D.Fix(pk, 2, 8, out FloatSpan2D kSpan);
                lSpan.ConvoluteGradient((3, 3, 2), kSpan, (2, 2, 2), out FloatSpan2D result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(expected));
                result.Free();
            }
        }
    }
}
