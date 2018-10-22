using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

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
        public static float[] Load<TPixel>([NotNull] string path, ImageNormalizationMode normalization, [CanBeNull] Action<IImageProcessingContext<TPixel>> modify) where TPixel : struct, IPixel<TPixel>
        {
            using (Image<TPixel> image = Image.Load<TPixel>(path))
            {
                if (modify != null) image.Mutate(modify);
                if (typeof(TPixel) == typeof(Alpha8)) return Load(image as Image<Alpha8>, normalization);
                if (typeof(TPixel) == typeof(Rgb24)) return Load(image as Image<Rgb24>, normalization);
                if (typeof(TPixel) == typeof(Argb32)) return Load(image as Image<Argb32>, normalization);
                if (typeof(TPixel) == typeof(Rgba32)) return Load(image as Image<Rgba32>, normalization);
                throw new InvalidOperationException($"The {typeof(TPixel).Name} pixel format isn't currently supported");
            }
        }

        #region Loaders

        // Loads an ARGB32 image
        [Pure, NotNull]
        private static unsafe float[] Load(Image<Argb32> image, ImageNormalizationMode normalization)
        {
            int resolution = image.Height * image.Width;
            float[] sample = new float[resolution * 4];
            fixed (Argb32* p0 = image.GetPixelSpan())
            fixed (float* psample = sample)
            {
                for (int i = 0; i < resolution; i++)
                {
                    Vector4 pixels = p0[i].Normalize(normalization);
                    psample[i] = pixels.W;
                    psample[i + resolution] = pixels.X;
                    psample[i + 2 * resolution] = pixels.Y;
                    psample[i + 3 * resolution] = pixels.Z;
                }
            }
            return sample;
        }

        // Loads an RGBA32 image
        [Pure, NotNull]
        private static unsafe float[] Load(Image<Rgba32> image, ImageNormalizationMode normalization)
        {
            int resolution = image.Height * image.Width;
            float[] sample = new float[resolution * 4];
            fixed (Rgba32* p0 = image.GetPixelSpan())
            fixed (float* psample = sample)
            {
                for (int i = 0; i < resolution; i++)
                {
                    Vector4 pixels = p0[i].Normalize(normalization);
                    psample[i] = pixels.X;
                    psample[i + resolution] = pixels.Y;
                    psample[i + 2 * resolution] = pixels.Z;
                    psample[i + 3 * resolution] = pixels.W;
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
            fixed (Rgb24* p0 = image.GetPixelSpan())
            fixed (float* psample = sample)
            {
                for (int i = 0; i < resolution; i++)
                {
                    Vector4 pixels = p0[i].Normalize(normalization);
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
            fixed (Alpha8* p0 = image.GetPixelSpan())
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

        /// <summary>
        /// Normalizes the input <see cref="IPixel"/> value using the specified mode
        /// </summary>
        /// <typeparam name="TPixel">Tye input pixel type</typeparam>
        /// <param name="pixel">The input pixel to normalize</param>
        /// <param name="normalization">The normalization mode to use</param>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector4 Normalize<TPixel>(this TPixel pixel, ImageNormalizationMode normalization) where TPixel : struct, IPixel<TPixel>
        {
            switch (normalization)
            {
                case ImageNormalizationMode.Sigmoid: return pixel.ToVector4(); // Already in the [0,1] range
                case ImageNormalizationMode.Normal: return Vector4.Subtract(pixel.ToVector4() * 2, Vector4.One);
                case ImageNormalizationMode.None: return pixel.ToVector4() * 255f; // Rescale in the [0,255] range
                default: throw new ArgumentOutOfRangeException(nameof(normalization), "Invalid normalization mode");
            }
        }
    }
}
