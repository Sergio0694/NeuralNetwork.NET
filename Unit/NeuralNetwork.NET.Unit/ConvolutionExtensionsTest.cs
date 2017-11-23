using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Helpers;

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
        public void Pool1()
        {
            // Test values
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
                },
                t = m.Pool2x2(1);
            Assert.IsTrue(t.ContentEquals(r));
            float[,]
                upscale = m.UpscalePool2x2(r, 1),
                expected =
                {
                    {
                        0, 0, 0, 2,
                        1, 0, 0, 0,
                        0, 0, 0, -0.5f,
                        0, 10, 0, 0
                    }
                };
            Assert.IsTrue(expected.ContentEquals(upscale));
        }

        [TestMethod]
        public void Pool2()
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
                },
                t = m.Pool2x2(1);
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Pool3()
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
                },
                t = m.Pool2x2(1);
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Pool4()
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
                },
                t = m.Pool2x2(1);
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Pool5()
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
                },
                t = m.Pool2x2(2);
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Rotate1()
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
                },
                t = m.Rotate180(1);
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Rotate2()
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
                },
                t = m.Rotate180(1);
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Rotate3()
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
                },
                t = m.Rotate180(2);
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void UpscalePool1()
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
                t = m.UpscalePool2x2(p, 1),
                r =
                {
                    {
                        0, 0, 0, 77,
                        66, 0, 0, 0,
                        0, 0, 0, 11,
                        0, 99, 0, 0
                    }
                };
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void UpscalePool2()
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
                t = m.UpscalePool2x2(p, 2),
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
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void UpscalePool3()
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
                t = m.UpscalePool2x2(p, 2),
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
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void UpscalePool4()
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
                t = m.UpscalePool2x2(p, 3),
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
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Compress1()
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
            float[]
                r = { 45 }, 
                t = m.CompressVertically(1);
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Compress2()
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
            float[]
                r = { 150, 227 },
                t = m.CompressVertically(2);
            Assert.IsTrue(t.ContentEquals(r));
        }

        // 1-depth, 3*3 with 2*2 = 2*2 result
        [TestMethod]
        public void Convolution2DValid1()
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
            float[,] result = l.ConvoluteForward(1, k, 1);
            float[,] expected =
            {
                {
                    2, 2,
                    4, 1
                }
            };
            Assert.IsTrue(result.ContentEquals(expected));
        }

        // 1-depth, 2 sample 3*3 with 2*2 = 2 sample 2*2 result
        [TestMethod]
        public void Convolution2DValid2()
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
            float[,] result = l.ConvoluteForward(1, k, 1);
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
            Assert.IsTrue(result.ContentEquals(expected));
        }

        // 1-depth, 3*3 with 2 kernels 2*2 = 2-depth 2*2 result
        [TestMethod]
        public void Convolution2DValid3()
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
            float[,] result = l.ConvoluteForward(1, k, 1);
            float[,] expected =
            {
                {
                    2, 2,
                    4, 1,

                    4, 0,
                    1, 3
                }
            };
            Assert.IsTrue(result.ContentEquals(expected));
        }

        // 2-depth, 3*3 with 2-depth kernel = 2*2 result
        [TestMethod]
        public void Convolution2DValid4()
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
            float[,] result = l.ConvoluteForward(2, k, 2);
            float[,] expected =
            {
                {
                    2, 4,
                    6, 3
                }
            };
            Assert.IsTrue(result.ContentEquals(expected));
        }

        [TestMethod]
        public void Convolution2DValid5()
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
            float[,] result = l.ConvoluteForward(2, k, 2);
            float[,] expected =
            {
                {
                    2, 4,
                    6, 3,

                    2, 4,
                    6, 3
                }
            };
            Assert.IsTrue(result.ContentEquals(expected));
        }

        [TestMethod]
        public void ConvolutionFull1()
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
            float[,] result = l.ConvoluteBackwards(1, k, 1);
            float[,] expected =
            {
                {
                    0, 1, 1,
                    -1, 1, 3,
                    0, -1, 2
                }
            };
            Assert.IsTrue(result.ContentEquals(expected));
        }

        [TestMethod]
        public void ConvolutionFull2()
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
            float[,] result = l.ConvoluteBackwards(2, k, 1);
            float[,] expected =
            {
                {
                    0, 0, 0,
                    -2, 2, 4,
                    0, -2, 4
                }
            };
            Assert.IsTrue(result.ContentEquals(expected));
        }

        [TestMethod]
        public void ConvolutionFull3()
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
            float[,] result = l.ConvoluteBackwards(1, k, 2);
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
            Assert.IsTrue(result.ContentEquals(expected));
        }

        [TestMethod]
        public void ConvolutionFull4()
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
            float[,] result = l.ConvoluteBackwards(2, k, 2);
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
            Assert.IsTrue(result.ContentEquals(expected));
        }

        [TestMethod]
        public void Convolution2DGradient1()
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
            float[,] result = l.ConvoluteGradient(1, k, 1);
            float[,] expected =
            {
                {
                    2, 2,
                    4, 1
                }
            };
            Assert.IsTrue(result.ContentEquals(expected));
        }
        
        [TestMethod]
        public void Convolution2DGradient2()
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
            float[,] result = l.ConvoluteGradient(1, k, 2);
            float[,] expected =
            {
                {
                    2, 2,
                    4, 1,

                    4, 2,
                    5, 2
                }
            };
            Assert.IsTrue(result.ContentEquals(expected));
        }

        [TestMethod]
        public void Convolution2DGradient3()
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
            float[,] result = l.ConvoluteGradient(2, k, 1);
            float[,] expected =
            {
                {
                    2, 2,
                    4, 1,

                    2, 2,
                    4, 1
                }
            };
            Assert.IsTrue(result.ContentEquals(expected));
        }

        [TestMethod]
        public void Convolution2DGradient4()
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
            float[,] result = l.ConvoluteGradient(2, k, 1);
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
            Assert.IsTrue(result.ContentEquals(expected));
        }

        [TestMethod]
        public void Convolution2DGradient5()
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
            float[,] result = l.ConvoluteGradient(2, k, 2);
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
            Assert.IsTrue(result.ContentEquals(expected));
        }

        [TestMethod]
        public void Sum1()
        {
            float[,]
                l =
                {
                    {
                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0
                    }
                };
            float[] v = { 1 };
            l.InPlaceSum(1, v);
            float[,] expected =
            {
                {
                    1, 2, 1,
                    3, 1, 2,
                    2, 2, 1
                }
            };
            Assert.IsTrue(l.ContentEquals(expected));
        }

        [TestMethod]
        public void Sum2()
        {
            float[,]
                l =
                {
                    {
                        0, 1, 0,
                        2, 0, 1,
                        1, 1, 0,

                        1, 2, 3,
                        4, 5, 6,
                        9, 8, 7
                    },
                    {
                        0, 1, 66,
                        2, 0, 199,
                        1, 1, 0,

                        1, 2, 3,
                        4, 5, 6,
                        9, 8, 7
                    }
                };
            float[] v = { 1, 2 };
            l.InPlaceSum(2, v);
            float[,] expected =
            {
                {
                    1, 2, 1,
                    3, 1, 2,
                    2, 2, 1,

                    3, 4, 5,
                    6, 7, 8,
                    11, 10, 9
                },
                {
                    1, 2, 67,
                    3, 1, 200,
                    2, 2, 1,

                    3, 4, 5,
                    6, 7, 8,
                    11, 10, 9
                }
            };
            Assert.IsTrue(l.ContentEquals(expected));
        }
    }
}
