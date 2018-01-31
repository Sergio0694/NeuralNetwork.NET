using System;
using System.Numerics;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
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
        /// <param name="normalization">The image normalization mode to apply</param>
        /// <param name="modify">The optional changes to apply to the image</param>
        [Pure, NotNull]
        public static float[] Load<TPixel>([NotNull] String path, ImageNormalizationMode normalization, [CanBeNull] Action<IImageProcessingContext<TPixel>> modify) where TPixel : struct, IPixel<TPixel>
        {
            using (Image<TPixel> image = Image.Load<TPixel>(path))
            {
                if (modify != null) image.Mutate(modify);
                if (typeof(TPixel) == typeof(Alpha8)) return Load(image as Image<Alpha8>, normalization);
                if (typeof(TPixel) == typeof(Rgb24)) return Load(image as Image<Rgb24>, normalization);
                if (typeof(TPixel) == typeof(Argb32)) return Load(image as Image<Argb32>, normalization);
                throw new InvalidOperationException($"The {typeof(TPixel).Name} pixel format isn't currently supported");
            }
        }

        #region Loaders

        // Loads an RGBA32 image
        [Pure, NotNull]
        private static unsafe float[] Load(Image<Argb32> image, ImageNormalizationMode normalization)
        {
            int resolution = image.Height * image.Width;
            float[] sample = new float[resolution * 4];
            fixed (Argb32* p0 = &image.DangerousGetPinnableReferenceToPixelBuffer())
            fixed (float* psample = sample)
            {
                for (int i = 0; i < resolution; i++)
                {
                    Vector4 pixels;
                    switch (normalization)
                    {
                        case ImageNormalizationMode.Sigmoid: pixels = p0[i].ToVector4(); break;
                        case ImageNormalizationMode.Normal: pixels = Vector4.Subtract(p0[i].ToVector4() * 2, Vector4.One); break;
                        case ImageNormalizationMode.None: pixels = new Vector4(p0[i].R, p0[i].G, p0[i].B, p0[i].A); break;
                        default: throw new ArgumentOutOfRangeException(nameof(normalization), "Invalid normalization mode");
                    }
                    psample[i] = pixels.W;
                    psample[i + resolution] = pixels.X;
                    psample[i + 2 * resolution] = pixels.Y;
                    psample[i + 3 * resolution] = pixels.Z;
                }
            }
            return sample;
        }

        // Loads an RGBA24 image
        [Pure, NotNull]
        private static unsafe float[] Load(Image<Rgb24> image, ImageNormalizationMode normalization)
        {
            int resolution = image.Height * image.Width;
            float[] sample = new float[resolution * 3];
            fixed (Rgb24* p0 = &image.DangerousGetPinnableReferenceToPixelBuffer())
            fixed (float* psample = sample)
            {
                for (int i = 0; i < resolution; i++)
                {
                    Vector4 pixels;
                    switch (normalization)
                    {
                        case ImageNormalizationMode.Sigmoid: pixels = p0[i].ToVector4(); break;
                        case ImageNormalizationMode.Normal: pixels = Vector4.Subtract(p0[i].ToVector4() * 2, Vector4.One); break;
                        case ImageNormalizationMode.None: pixels = new Vector4(p0[i].R, p0[i].G, p0[i].B, 255f); break;
                        default: throw new ArgumentOutOfRangeException(nameof(normalization), "Invalid normalization mode");
                    }
                    psample[i] = pixels.X;
                    psample[i + resolution] = pixels.Y;
                    psample[i + 2 * resolution] = pixels.Z;
                }
            }
            return sample;
        }

        // Loads an ALPHA8 image
        [Pure, NotNull]
        private static unsafe float[] Load(Image<Alpha8> image, ImageNormalizationMode normalization)
        {
            int resolution = image.Height * image.Width;
            float[] sample = new float[resolution];
            fixed (Alpha8* p0 = &image.DangerousGetPinnableReferenceToPixelBuffer())
            fixed (float* psample = sample)
                for (int i = 0; i < resolution; i++)
                {
                    switch (normalization)
                    {
                        case ImageNormalizationMode.Sigmoid: psample[i] = p0[i].PackedValue / 255f; break;
                        case ImageNormalizationMode.Normal: psample[i] = p0[i].PackedValue * 2 / 255f - 1; break;
                        case ImageNormalizationMode.None: psample[i] = p0[i].PackedValue; break;
                        default: throw new ArgumentOutOfRangeException(nameof(normalization), "Invalid normalization mode");
                    }
                }
            return sample;
        }

        #endregion
    }
}
