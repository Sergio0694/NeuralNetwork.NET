namespace ConvolutionalNeuralNetworkLibrary
{
    /// <summary>
    /// A class that contains a collection of 3x3 kernels
    /// </summary>
    public static class KernelsCollection
    {
        #region Edge detection

        /// <summary>
        /// { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 }
        /// </summary>
        public static double[,] TopSobel { get; } = new double[,]
        {
            { 1, 2, 1 },
            { 0, 0, 0 },
            { -1, -2, -1 }
        };

        /// <summary>
        /// { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 }
        /// </summary>
        public static double[,] BottomSobel { get; } = new double[,]
        {
            { -1, -2, -1 },
            { 0, 0, 0 },
            { 1, 2, 1 }
        };

        /// <summary>
        /// { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 }
        /// </summary>
        public static double[,] LeftSobel { get; } = new double[,]
        {
            { 1, 0, -1 },
            { 2, 0, -2 },
            { 1, 0, -1 }
        };

        /// <summary>
        /// { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 }
        /// </summary>
        public static double[,] RightSobel { get; } = new double[,]
        {
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
        };

        /// <summary>
        /// { 0, -1, 0 }, { 0, 2, 0 }, { 0, -1, 0 }
        /// </summary>
        public static double[,] VerticalSobel { get; } = new double[,]
        {
            { 0, -1, 0 },
            { 0, 2, 0 },
            { 0, -1, 0 }
        };

        /// <summary>
        /// { 0, 0, 0 }, { -1, 2, -1 }, { 0, 0, 0 }
        /// </summary>
        public static double[,] HorizontalSobel { get; } = new double[,]
        {
            { 0, 0, 0 },
            { -1, 2, -1 },
            { 0, 0, 0 }
        };

        #endregion

        /// <summary>
        /// Gets a sharpening kernel with a maximum value of 5
        /// </summary>
        public static double[,] Sharpen { get; } = new double[,]
        {
            { 0, -1, 0 },
            { -1, 5, -1 },
            { 0, -1, 0 }
        };

        /// <summary>
        /// Gets an outline kernel with a maximum value of 8
        /// </summary>
        public static double[,] Outline { get; } = new double[,]
        {
            { -1, -1, -1 },
            { -1, 2, -1 },
            { -1, -1, -1 }
        };

        #region Emboss

        /// <summary>
        /// { -2, -1, 0 }, { -1, 1, 1 }, { 0, 1, 2 }
        /// </summary>
        public static double[,] BottomRightEmboss { get; } = new double[,]
        {
            { -2, -1, 0 },
            { -1, 1, 1 },
            { 0, 1, 2 }
        };

        /// <summary>
        /// { 0, 1, 2 }, { -1, 1, 1 }, { -2, -1, 0 }
        /// </summary>
        public static double[,] TopRightEmboss { get; } = new double[,]
        {
            { 0, 1, 2 },
            { -1, 1, 1 },
            { -2, -1, 0 }
        };

        /// <summary>
        /// { 2, 1, 0 }, { 1, 1, -1 }, { 0, -1, -2 }
        /// </summary>
        public static double[,] TopLeftEmboss { get; } = new double[,]
        {
            { 2, 1, 0 },
            { 1, 1, -1 },
            { 0, -1, -2 }
        };

        /// <summary>
        /// { 0, -1, -2 }, { 1, 1, -1 }, { 2, 1, 0 }
        /// </summary>
        public static double[,] BottomLeftEmboss { get; } = new double[,]
        {
            { 0, -1, -2 },
            { 1, 1, -1 },
            { 2, 1, 0 }
        };

        #endregion

        #region Kirsch kernel

        /// <summary>
        /// N
        /// </summary>
        public static double[,] KirschG1 { get; } = new double[,]
        {
            { 5, 5, 5 },
            { -3, 0, -3 },
            { -3, -3, -3 }
        };

        /// <summary>
        /// NW
        /// </summary>
        public static double[,] KirschG2 { get; } = new double[,]
        {
            { 5, 5, -3 },
            { 5, 0, -3 },
            { -3, -3, -3 }
        };

        /// <summary>
        /// W
        /// </summary>
        public static double[,] KirschG3 { get; } = new double[,]
        {
            { 5, -3, -3 },
            { 5, 0, -3 },
            { 5, -3, -3 }
        };

        /// <summary>
        /// SW
        /// </summary>
        public static double[,] KirschG4 { get; } = new double[,]
        {
            { -3, -3, -3 },
            { 5, 0, -3 },
            { 5, 5, -3 }
        };

        /// <summary>
        /// S
        /// </summary>
        public static double[,] KirschG5 { get; } = new double[,]
        {
            { -3, -3, -3 },
            { -3, 0, -3 },
            { 5, 5, 5 }
        };

        /// <summary>
        /// SE
        /// </summary>
        public static double[,] KirschG6 { get; } = new double[,]
        {
            { -3, -3, -3 },
            { -3, 0, 5 },
            { -3, 5, 5 }
        };

        /// <summary>
        /// E
        /// </summary>
        public static double[,] KirschG7 { get; } = new double[,]
        {
            { -3, -3, 5 },
            { -3, 0, 5 },
            { -3, -3, 5 }
        };

        /// <summary>
        /// NE
        /// </summary>
        public static double[,] KirschG8 { get; } = new double[,]
        {
            { -3, 5, 5 },
            { -3, 0, 5 },
            { -3, -3, -3 }
        };

        #endregion
    }
}
