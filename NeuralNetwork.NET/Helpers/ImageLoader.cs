using System;
using JetBrains.Annotations;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A static class with some helper methods to quickly load a sample from a target image file
    /// </summary>
    internal static class ImageLoader
    {
        /// <summary>
        /// Loads the target image and applies the requested changes, then converts it to a dataset sample
        /// </summary>
        /// <param name="path">The path of the image to load</param>
        /// <param name="modify">The optional changes to apply to the image</param>
        [Pure, NotNull]
        public static float[] Load<TPixel>([NotNull] String path, [CanBeNull] Action<IImageProcessingContext<TPixel>> modify) where TPixel : struct, IPixel<TPixel>
        {
            using (Image<TPixel> image = Image.Load<TPixel>(path))
            {
                if (modify != null) image.Mutate(modify);
                if (typeof(TPixel) == typeof(Alpha8)) return Load(image as Image<Alpha8>);
                if (typeof(TPixel) == typeof(Rgb24)) return Load(image as Image<Rgb24>);
                if (typeof(TPixel) == typeof(Argb32)) return Load(image as Image<Argb32>);
                throw new InvalidOperationException($"The {typeof(TPixel).Name} pixel format isn't currently supported");
            }
        }

        #region Loaders

        // Loads an RGBA32 image
        [Pure, NotNull]
        private static unsafe float[] Load(Image<Argb32> image)
        {
            int resolution = image.Height * image.Width;
            float[] sample = new float[resolution * 4];
            fixed (Argb32* p0 = &image.DangerousGetPinnableReferenceToPixelBuffer())
            fixed (float* psample = sample)
            {
                for (int i = 0; i < resolution; i++)
                {
                    Argb32* pxy = p0 + i;
                    psample[i] = pxy->A / 255f;
                    psample[i + resolution] = pxy->R / 255f;
                    psample[i + 2 * resolution] = pxy->G / 255f;
                    psample[i + 3 * resolution] = pxy->B / 255f;
                }
            }
            return sample;
        }

        // Loads an RGBA24 image
        [Pure, NotNull]
        private static unsafe float[] Load(Image<Rgb24> image)
        {
            int resolution = image.Height * image.Width;
            float[] sample = new float[resolution * 3];
            fixed (Rgb24* p0 = &image.DangerousGetPinnableReferenceToPixelBuffer())
            fixed (float* psample = sample)
            {
                for (int i = 0; i < resolution; i++)
                {
                    Rgb24* pxy = p0 + i;
                    psample[i] = pxy->R / 255f;
                    psample[i + resolution] = pxy->G / 255f;
                    psample[i + 2 * resolution] = pxy->B / 255f;
                }
            }
            return sample;
        }

        // Loads an ALPHA8 image
        [Pure, NotNull]
        private static unsafe float[] Load(Image<Alpha8> image)
        {
            int resolution = image.Height * image.Width;
            float[] sample = new float[resolution];
            fixed (Alpha8* p0 = &image.DangerousGetPinnableReferenceToPixelBuffer())
            fixed (float* psample = sample)
                for (int i = 0; i < resolution; i++)
                    psample[i] = p0[i].PackedValue / 255f;
            return sample;
        }

        #endregion
    }
}
