using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkDotNet.Core.Structs;
using NeuralNetworkDotNet.Cpu.cpuDNN;

namespace NeuralNetwork.NET.Cpu.Unit
{
    /// <summary>
    /// Test class for the convolution primitives
    /// </summary>
    [TestClass]
    [TestCategory("CpuDnnTests.Pooling")]
    public class CpuDnnTests_Poooling
    {
        [TestMethod]
        public void PoolingForward1()
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
                r1 =
                {
                    {
                        1, 2,
                        10, -0.5f
                    }
                },
                r2 =
                {
                    {
                        0, 0, 0, 2,
                        1, 0, 0, 0,
                        0, 0, 0, -0.5f,
                        0, 10, 0, 0
                    }
                };

            using (var tm = Tensor.From(m, 1, 4, 4))
            using (var tr1 = Tensor.From(r1, 1, 2, 2))
            using (var tr2 = Tensor.From(r2, 1, 4, 4))
            using (var y = Tensor.Like(tr1))
            {
                CpuDnn.PoolingForward(tm, y);
                Assert.IsTrue(y.Equals(tr1));

                // Upscale
                CpuDnn.PoolingBackward(tm, y, tm);
                Assert.IsTrue(tm.Equals(tr2));
            }
        }

        [TestMethod]
        public void PoolingForward2()
        {
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

            using (var tm = Tensor.From(m, 1, 7, 7))
            using (var tr = Tensor.From(r, 1, 4, 4))
            using (var y = Tensor.Like(tr))
            {
                CpuDnn.PoolingForward(tm, y);
                Assert.IsTrue(y.Equals(tr));
            }
        }

        [TestMethod]
        public void PoolingForward3()
        {
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

            using (var tm = Tensor.From(m, 1, 2, 2))
            using (var tr = Tensor.From(r, 1, 1, 1))
            using (var y = Tensor.Like(tr))
            {
                CpuDnn.PoolingForward(tm, y);
                Assert.IsTrue(y.Equals(tr));
            }
        }

        [TestMethod]
        public void PoolingForward4()
        {
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

            using (var tm = Tensor.From(m, 1, 4, 4))
            using (var tr = Tensor.From(r, 1, 2, 2))
            using (var y = Tensor.Like(tr))
            {
                CpuDnn.PoolingForward(tm, y);
                Assert.IsTrue(y.Equals(tr));
            }
        }

        [TestMethod]
        public void PoolingForward5()
        {
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

            using (var tm = Tensor.From(m, 2, 4, 4))
            using (var tr = Tensor.From(r, 2, 2, 2))
            using (var y = Tensor.Like(tr))
            {
                CpuDnn.PoolingForward(tm, y);
                Assert.IsTrue(y.Equals(tr));
            }
        }

        [TestMethod]
        public void PoolingBackward1()
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

            using (var tm = Tensor.From(m, 1, 4, 4))
            using (var tp = Tensor.From(p, 1, 2, 2))
            using (var tr = Tensor.From(r, 1, 4, 4))
            {
                CpuDnn.PoolingBackward(tm, tp, tm);
                Assert.IsTrue(tm.Equals(tr));
            }
        }

        [TestMethod]
        public void PoolingBackward2()
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

            using (var tm = Tensor.From(m, 2, 4, 4))
            using (var tp = Tensor.From(p, 2, 2, 2))
            using (var tr = Tensor.From(r, 2, 4, 4))
            {
                CpuDnn.PoolingBackward(tm, tp, tm);
                Assert.IsTrue(tm.Equals(tr));
            }
        }

        [TestMethod]
        public void PoolingBackward3()
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

            using (var tm = Tensor.From(m, 2, 4, 4))
            using (var tp = Tensor.From(p, 2, 2, 2))
            using (var tr = Tensor.From(r, 2, 4, 4))
            {
                CpuDnn.PoolingBackward(tm, tp, tm);
                Assert.IsTrue(tm.Equals(tr));
            }
        }

        [TestMethod]
        public void PoolingBackward4()
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

            using (var tm = Tensor.From(m, 3, 4, 4))
            using (var tp = Tensor.From(p, 3, 2, 2))
            using (var tr = Tensor.From(r, 3, 4, 4))
            {
                CpuDnn.PoolingBackward(tm, tp, tm);
                Assert.IsTrue(tm.Equals(tr));
            }
        }
    }
}
