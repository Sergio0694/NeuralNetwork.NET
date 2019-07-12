using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkDotNet.Core.Enums;
using NeuralNetworkDotNet.Core.Structs;
using NeuralNetworkDotNet.Cpu.cpuDNN;

namespace NeuralNetwork.NET.Cpu.Unit
{
    /// <summary>
    /// Test class for the convolution primitives
    /// </summary>
    [TestClass]
    [TestCategory("CpuDnnTests.Convolution")]
    public class CpuDnnTests_Convolution
    {
        // 1-depth, 3*3 with 2*2 = 2*2 result
        [TestMethod]
        public void ConvolutionForward1()
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
            float[,] r =
            {
                {
                    2.6f, 2.6f,
                    4.6f, 1.6f
                }
            };

            using (var tl = Tensor.From(l, 1, 3, 3))
            using (var tk = Tensor.From(k, 1, 2, 2))
            using (var tb = Tensor.From(b))
            using (var tr = Tensor.From(r, 1, 2, 2))
            using (var y = Tensor.Like(tr))
            {
                CpuDnn.ConvolutionForward(tl, tk, tb, y);
                Assert.IsTrue(y.Equals(tr));
            }
        }

        // 1-depth, 2 sample 3*3 with 2*2 = 2 sample 2*2 result
        [TestMethod]
        public void ConvolutionForward2()
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
                },
                r =
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

            using (var tl = Tensor.From(l, 1, 3, 3))
            using (var tk = Tensor.From(k, 1, 2, 2))
            using (var tr = Tensor.From(r, 1, 2, 2))
            using (var tb = Tensor.New(1, 1, AllocationMode.Clean))
            using (var y = Tensor.Like(tr))
            {
                CpuDnn.ConvolutionForward(tl, tk, tb, y);
                Assert.IsTrue(y.Equals(tr));
            }
        }

        // 1-depth, 3*3 with 2 kernels 2*2 = 2-depth 2*2 result
        [TestMethod]
        public void ConvolutionForward3()
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
            float[] b = { 1, 0.5f };
            float[,] r =
            {
                {
                    3, 3,
                    5, 2,

                    4.5f, 0.5f,
                    1.5f, 3.5f
                }
            };

            using (var tl = Tensor.From(l, 1, 3, 3))
            using (var tk = Tensor.From(k, 1, 2, 2))
            using (var tb = Tensor.From(b))
            using (var tr = Tensor.From(r, 2, 2, 2))
            using (var y = Tensor.Like(tr))
            {
                CpuDnn.ConvolutionForward(tl, tk, tb, y);
                Assert.IsTrue(y.Equals(tr));
            }
        }

        // 2-depth, 3*3 with 2-depth kernel = 2*2 result
        [TestMethod]
        public void ConvolutionForward4()
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
            float[] b = { 0.1f };
            float[,] r =
            {
                {
                    2.1f, 4.1f,
                    6.1f, 3.1f
                }
            };

            using (var tl = Tensor.From(l, 2, 3, 3))
            using (var tk = Tensor.From(k, 2, 2, 2))
            using (var tb = Tensor.From(b))
            using (var tr = Tensor.From(r, 1, 2, 2))
            using (var y = Tensor.Like(tr))
            {
                CpuDnn.ConvolutionForward(tl, tk, tb, y);
                Assert.IsTrue(y.Equals(tr));
            }
        }

        [TestMethod]
        public void ConvolutionForward5()
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
            float[] b = { 0, 0.2f };
            float[,] r =
            {
                {
                    2, 4,
                    6, 3,

                    2.2f, 4.2f,
                    6.2f, 3.2f
                }
            };

            using (var tl = Tensor.From(l, 2, 3, 3))
            using (var tk = Tensor.From(k, 2, 2, 2))
            using (var tb = Tensor.From(b))
            using (var tr = Tensor.From(r, 2, 2, 2))
            using (var y = Tensor.Like(tr))
            {
                CpuDnn.ConvolutionForward(tl, tk, tb, y);
                Assert.IsTrue(y.Equals(tr));
            }
        }

        [TestMethod]
        public void ConvolutionBackwardData1()
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
                        1, 0,
                        1, 1
                    }
                };
            float[,] r =
            {
                {
                    0, 1, 1,
                    -1, 1, 3,
                    0, -1, 2
                }
            };

            using (var tl = Tensor.From(l, 1, 2, 2))
            using (var tk = Tensor.From(k, 1, 2, 2))
            using (var tr = Tensor.From(r, 1, 3, 3))
            using (var y = Tensor.Like(tr))
            {
                CpuDnn.ConvolutionBackwardData(tl, tk, y);
                Assert.IsTrue(y.Equals(tr));
            }
        }

        [TestMethod]
        public void ConvolutionBackwardData2()
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
                        1, 0,
                        1, 1
                    },
                    {
                        1, 0,
                        1, 1
                    }
                };
            float[,] r =
            {
                {
                    0, 0, 0,
                    -2, 2, 4,
                    0, -2, 4
                }
            };

            using (var tl = Tensor.From(l, 2, 2, 2))
            using (var tk = Tensor.From(k, 1, 2, 2))
            using (var tr = Tensor.From(r, 1, 3, 3))
            using (var y = Tensor.Like(tr))
            {
                CpuDnn.ConvolutionBackwardData(tl, tk, y);
                Assert.IsTrue(y.Equals(tr));
            }
        }

        [TestMethod]
        public void ConvolutionBackwardData3()
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
                        1, 0,
                        1, 1,

                        0, 1,
                        0, 1
                    }
                };
            float[,] r =
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

            using (var tl = Tensor.From(l, 1, 2, 2))
            using (var tk = Tensor.From(k, 2, 2, 2))
            using (var tr = Tensor.From(r, 2, 3, 3))
            using (var y = Tensor.Like(tr))
            {
                CpuDnn.ConvolutionBackwardData(tl, tk, y);
                Assert.IsTrue(y.Equals(tr));
            }
        }

        [TestMethod]
        public void ConvolutionBackwardData4()
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
                        1, 0,
                        1, 1,

                        0, 1,
                        0, 1
                    },
                    {
                        1, 0,
                        0, 1,

                        1, 3,
                        0, -2
                    }
                };
            float[,] r =
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

            using (var tl = Tensor.From(l, 2, 2, 2))
            using (var tk = Tensor.From(k, 2, 2, 2))
            using (var tr = Tensor.From(r, 2, 3, 3))
            using (var y = Tensor.Like(tr))
            {
                CpuDnn.ConvolutionBackwardData(tl, tk, y);
                Assert.IsTrue(y.Equals(tr));
            }
        }

        [TestMethod]
        public void ConvolutionBackwardBias1()
        {
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

            using (var tm = Tensor.From(m, 1, 3, 3))
            using (var tr = Tensor.From(r))
            using (var y = Tensor.Like(tr))
            {
                CpuDnn.ConvolutionBackwardBias(tm, y);
                Assert.IsTrue(y.Equals(tr));
            }
        }

        [TestMethod]
        public void ConvolutionBackwardBias2()
        {
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

            using (var tm = Tensor.From(m, 2, 3, 3))
            using (var tr = Tensor.From(r, 1, 2, 1, 1))
            using (var y = Tensor.Like(tr))
            {
                CpuDnn.ConvolutionBackwardBias(tm, y);
                Assert.IsTrue(y.Equals(tr));
            }
        }
    }
}
