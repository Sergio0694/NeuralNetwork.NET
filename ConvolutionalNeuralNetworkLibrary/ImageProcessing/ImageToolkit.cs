using System.Drawing;
using System.Drawing.Imaging;
using JetBrains.Annotations;

namespace ConvolutionalNeuralNetworkLibrary.ImageProcessing
{
    /// <summary>
    /// A static class with some helper methods to manipulate images
    /// </summary>
    public static class ImageToolkit
    {
        /// <summary>
        /// Extracts the pixel data from an input <see cref="Bitmap"/> and normalizes the values in the [0..1] range
        /// </summary>
        /// <param name="image">The source <see cref="Bitmap"/> image</param>
        /// <remarks>Only the R color channel is used, as the input image is supposed to be in grayscale</remarks>
        [PublicAPI]
        [Pure]
        [NotNull]
        public static double[,] ToNormalizedPixelData([NotNull] this Bitmap image)
        {
            double[,] normalized = new double[image.Height, image.Width];
            for (int i = 0; i < image.Height; i++)
                for (int j = 0; j < image.Width; j++)
                    normalized[i, j] = (byte.MaxValue - image.GetPixel(i, j).R) / (double)byte.MaxValue;
            return normalized;
        }

        /// <summary>
        /// Converts the input <see cref="Bitmap"/> image to a grayscale image
        /// </summary>
        /// <param name="original">The <see cref="Bitmap"/> image to convert</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        public static Bitmap ToGrayscale([NotNull] this Bitmap original)
        {
            // Create a blank bitmap the same size as original
            Bitmap newBitmap = new Bitmap(original.Width, original.Height);

            // Get a graphics object from the new image
            using (Graphics g = Graphics.FromImage(newBitmap))
            {
                // Create the grayscale ColorMatrix
                ColorMatrix colorMatrix = new ColorMatrix(
                new float[][]
                {
                    new float[] { 0.3f, 0.3f, 0.3f, 0, 0 },
                    new float[] { 0.59f, 0.59f, 0.59f, 0, 0 },
                    new float[] { 0.11f, 0.11f, 0.11f, 0, 0 },
                    new float[] { 0, 0, 0, 1, 0 },
                    new float[] { 0, 0, 0, 0, 1 }
                });

                // Create the image attributes and set the color matrix
                ImageAttributes attributes = new ImageAttributes();
                attributes.SetColorMatrix(colorMatrix);

                // Draw the original image on the new image using the grayscale color matrix
                g.DrawImage(original, new Rectangle(0, 0, original.Width, original.Height),
                    0, 0, original.Width, original.Height, GraphicsUnit.Pixel, attributes);
                return newBitmap;
            }
        }
    }
}
