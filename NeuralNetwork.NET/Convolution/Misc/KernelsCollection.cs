namespace NeuralNetworkNET.Convolution.Misc
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
        public static float[,] TopSobel { get; } =
        {
            { 1, 2, 1 },
            { 0, 0, 0 },
            { -1, -2, -1 }
        };

        /// <summary>
        /// { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 }
        /// </summary>
        public static float[,] BottomSobel { get; } =
        {
            { -1, -2, -1 },
            { 0, 0, 0 },
            { 1, 2, 1 }
        };

        /// <summary>
        /// { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 }
        /// </summary>
        public static float[,] LeftSobel { get; } =
        {
            { 1, 0, -1 },
            { 2, 0, -2 },
            { 1, 0, -1 }
        };

        /// <summary>
        /// { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 }
        /// </summary>
        public static float[,] RightSobel { get; } =
        {
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
        };

        /// <summary>
        /// { 0, -1, 0 }, { 0, 2, 0 }, { 0, -1, 0 }
        /// </summary>
        public static float[,] VerticalSobel { get; } =
        {
            { 0, -1, 0 },
            { 0, 2, 0 },
            { 0, -1, 0 }
        };

        /// <summary>
        /// { 0, 0, 0 }, { -1, 2, -1 }, { 0, 0, 0 }
        /// </summary>
        public static float[,] HorizontalSobel { get; } =
        {
            { 0, 0, 0 },
            { -1, 2, -1 },
            { 0, 0, 0 }
        };

        #endregion

        /// <summary>
        /// Gets a sharpening kernel with a maximum value of 5
        /// </summary>
        public static float[,] Sharpen { get; } =
        {
            { 0, -1, 0 },
            { -1, 5, -1 },
            { 0, -1, 0 }
        };

        /// <summary>
        /// Gets an outline kernel with a maximum value of 8
        /// </summary>
        public static float[,] Outline { get; } =
        {
            { -1, -1, -1 },
            { -1, 2, -1 },
            { -1, -1, -1 }
        };

        #region Emboss

        /// <summary>
        /// { -2, -1, 0 }, { -1, 1, 1 }, { 0, 1, 2 }
        /// </summary>
        public static float[,] BottomRightEmboss { get; } =
        {
            { -2, -1, 0 },
            { -1, 1, 1 },
            { 0, 1, 2 }
        };

        /// <summary>
        /// { 0, 1, 2 }, { -1, 1, 1 }, { -2, -1, 0 }
        /// </summary>
        public static float[,] TopRightEmboss { get; } =
        {
            { 0, 1, 2 },
            { -1, 1, 1 },
            { -2, -1, 0 }
        };

        /// <summary>
        /// { 2, 1, 0 }, { 1, 1, -1 }, { 0, -1, -2 }
        /// </summary>
        public static float[,] TopLeftEmboss { get; } =
        {
            { 2, 1, 0 },
            { 1, 1, -1 },
            { 0, -1, -2 }
        };

        /// <summary>
        /// { 0, -1, -2 }, { 1, 1, -1 }, { 2, 1, 0 }
        /// </summary>
        public static float[,] BottomLeftEmboss { get; } =
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
        public static float[,] KirschG1 { get; } =
        {
            { 5, 5, 5 },
            { -3, 0, -3 },
            { -3, -3, -3 }
        };

        /// <summary>
        /// NW
        /// </summary>
        public static float[,] KirschG2 { get; } =
        {
            { 5, 5, -3 },
            { 5, 0, -3 },
            { -3, -3, -3 }
        };

        /// <summary>
        /// W
        /// </summary>
        public static float[,] KirschG3 { get; } =
        {
            { 5, -3, -3 },
            { 5, 0, -3 },
            { 5, -3, -3 }
        };

        /// <summary>
        /// SW
        /// </summary>
        public static float[,] KirschG4 { get; } =
        {
            { -3, -3, -3 },
            { 5, 0, -3 },
            { 5, 5, -3 }
        };

        /// <summary>
        /// S
        /// </summary>
        public static float[,] KirschG5 { get; } =
        {
            { -3, -3, -3 },
            { -3, 0, -3 },
            { 5, 5, 5 }
        };

        /// <summary>
        /// SE
        /// </summary>
        public static float[,] KirschG6 { get; } =
        {
            { -3, -3, -3 },
            { -3, 0, 5 },
            { -3, 5, 5 }
        };

        /// <summary>
        /// E
        /// </summary>
        public static float[,] KirschG7 { get; } =
        {
            { -3, -3, 5 },
            { -3, 0, 5 },
            { -3, -3, 5 }
        };

        /// <summary>
        /// NE
        /// </summary>
        public static float[,] KirschG8 { get; } =
        {
            { -3, 5, 5 },
            { -3, 0, 5 },
            { -3, -3, -3 }
        };

        #endregion
    }
}
