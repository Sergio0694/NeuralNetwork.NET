using System;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Helpers;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.Primitives;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A static class with some helper methods to quickly load a sample from a target image file
    /// </summary>
    public static class ImageLoader
    {
        /// <summary>
        /// Loads the target image and applies the requested changes, then converts it to a dataset sample
        /// </summary>
        /// <param name="path">The path of the image to load</param>
        /// <param name="modify">The optional changes to apply to the image</param>
        [PublicAPI]
        [Pure, NotNull]
        public static unsafe float[] LoadRgb32ImageSample([NotNull] String path, [CanBeNull] Action<IImageProcessingContext<Rgb24>> modify = null)
        {
            using (Image<Rgb24> image = Image.Load<Rgb24>(path))
            {
                if (modify != null) image.Mutate(modify);
                Size size = image.Size();
                ref Rgb24 pixel0 = ref image.DangerousGetPinnableReferenceToPixelBuffer();
                Rgb24* p0 = (Rgb24*)Unsafe.AsPointer(ref pixel0);
                int resolution = size.Height * size.Width;
                float[] sample = new float[resolution * 3];
                fixed (float* psample = sample)
                    for (int i = 0; i < resolution; i++)
                    {
                        Rgb24* pxy = p0 + i;
                        psample[i] = pxy->R;
                        psample[i + resolution] = pxy->G;
                        psample[i + 2 * resolution] = pxy->B;
                    }
                return sample;
            }
        }

        /// <summary>
        /// Loads the target image, converts it to grayscale and applies the requested changes, then converts it to a dataset sample
        /// </summary>
        /// <param name="path">The path of the image to load</param>
        /// <param name="modify">The optional changes to apply to the image</param>
        [PublicAPI]
        [Pure, NotNull]
        public static unsafe float[] LoadGrayscaleImageSample([NotNull] String path, [CanBeNull] Action<IImageProcessingContext<Rgb24>> modify = null)
        {
            using (Image<Rgb24> image = Image.Load<Rgb24>(path))
            {
                image.Mutate(x => x.Grayscale());
                if (modify != null) image.Mutate(modify);
                Size size = image.Size();
                ref Rgb24 pixel0 = ref image.DangerousGetPinnableReferenceToPixelBuffer();
                Rgb24* p0 = (Rgb24*)Unsafe.AsPointer(ref pixel0);
                int resolution = size.Height * size.Width;
                float[] sample = new float[resolution];
                fixed (float* psample = sample)
                    for (int i = 0; i < resolution; i++)
                        psample[i] = p0[i].R;
                return sample;
            }
        }
    }
}
