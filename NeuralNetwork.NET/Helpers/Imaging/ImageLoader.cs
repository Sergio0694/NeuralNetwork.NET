using System;
using System.IO;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.Primitives;

namespace NeuralNetworkNET.Helpers.Imaging
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
                int resolution = image.Height * image.Width;
                float[] sample = new float[resolution * 3];
                fixed (Rgb24* p0 = &image.DangerousGetPinnableReferenceToPixelBuffer())
                fixed (float* psample = sample)
                    for (int i = 0; i < resolution; i++)
                    {
                        Rgb24* pxy = p0 + i;
                        psample[i] = pxy->R / 255f;
                        psample[i + resolution] = pxy->G / 255f;
                        psample[i + 2 * resolution] = pxy->B / 255f;
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
                int resolution = image.Height * image.Width;
                float[] sample = new float[resolution];
                fixed (Rgb24* p0 = &image.DangerousGetPinnableReferenceToPixelBuffer())
                fixed (float* psample = sample)
                    for (int i = 0; i < resolution; i++)
                        psample[i] = p0[i].R / 255f;
                return sample;
            }
        }

        /// <summary>
        /// Saves an image in the target path representing the input weights and biases
        /// </summary>
        /// <param name="path">The target path for the image</param>
        /// <param name="weights">The input weights</param>
        /// <param name="biases">The input biases</param>
        /// <param name="scaling">The desired image scaling to use</param>
        [PublicAPI]
        public static unsafe void ExportFullyConnectedWeights([NotNull] String path, [NotNull] float[,] weights, [NotNull] float[] biases, ImageScaling scaling)
        {
            int
                h = weights.GetLength(0),
                w = weights.GetLength(1);
            if (biases.Length != w) throw new ArgumentException("The biases length must match the width of the weights matrix");
            using (Image<Rgb24> image = new Image<Rgb24>(w, h + 1))
            {
                fixed (Rgb24* p0 = &image.DangerousGetPinnableReferenceToPixelBuffer())
                {
                    // Weights
                    fixed (float* pw = weights)
                    {
                        (float min, float max) = MatrixExtensions.MinMax(new ReadOnlySpan<float>(pw, h * w));
                        for (int i = 0; i < h; i++)
                        {
                            int offset = i * w;
                            for (int j = 0; j < w; j++)
                            {
                                float
                                    value = pw[offset + j],
                                    normalized = (value - min) * 255 / (max - min);
                                byte hex = (byte)normalized;
                                p0[j * h + i] = new Rgb24(hex, hex, hex);
                            }
                        }
                    }

                    // Biases
                    fixed (float* pb = biases)
                    {
                        (float min, float max) = MatrixExtensions.MinMax(new ReadOnlySpan<float>(pb, w));
                        for (int i = 0; i < w; i++)
                        {
                            float
                                value = pb[i],
                                normalized = (value - min) * 255 / (max - min);
                            byte hex = (byte)normalized;
                            p0[h * w + i] = new Rgb24(hex, hex, hex);
                        }
                    }
                }
                image.UpdateScaling(scaling);
                using (FileStream stream = File.OpenWrite(path.EndsWith(".png") ? path : $"{path}.png"))
                    image.Save(stream, ImageFormats.Png);
            }
        }

        /// <summary>
        /// Saves aan image for each kernel to the target directory
        /// </summary>
        /// <param name="directory">The directory to use to store the images</param>
        /// <param name="kernels">The input kernels</param>
        /// <param name="kernelsInfo">The size info for the input kernels</param>
        /// <param name="scaling">The desired image scaling to use</param>
        [PublicAPI]
        public static unsafe void ExportGrayscaleKernels([NotNull] String directory, [NotNull] float[,] kernels, TensorInfo kernelsInfo, ImageScaling scaling)
        {
            // Setup
            Directory.CreateDirectory(directory);
            int
                h = kernels.GetLength(0),
                w = kernels.GetLength(1);
            if (kernelsInfo.Channels != 1) throw new ArgumentException("Only a 2D kernel can be exported as an image with this method");

            // Export a single kernel matrix (one per weights row)
            void Kernel(int k)
            {
                using (Image<Rgb24> image = new Image<Rgb24>(kernelsInfo.Width, kernelsInfo.Height))
                {
                    fixed (Rgb24* p0 = &image.DangerousGetPinnableReferenceToPixelBuffer())
                    fixed (float* pw = kernels)
                    {
                        float* pwoffset = pw + k * w;
                        (float min, float max) = MatrixExtensions.MinMax(new ReadOnlySpan<float>(pwoffset, kernelsInfo.SliceSize));
                        for (int i = 0; i < kernelsInfo.Height; i++)
                        {
                            int offset = i * kernelsInfo.Width;
                            for (int j = 0; j < kernelsInfo.Width; j++)
                            {
                                float
                                    value = pwoffset[offset + j],
                                    normalized = (value - min) * 255 / (max - min);
                                byte hex = (byte)normalized;
                                p0[j * kernelsInfo.Height + i] = new Rgb24(hex, hex, hex);
                            }
                        }
                    }
                    image.UpdateScaling(scaling);
                    using (FileStream stream = File.OpenWrite(Path.Combine(directory, $"{k}.png")))
                        image.Save(stream, ImageFormats.Png);
                }
            }

            // Save all the kernels in parallel
            Parallel.For(0, h, Kernel).AssertCompleted();
        }

        // Resizes an image with the NN samples and the desired scaling mode
        private static void UpdateScaling<TPixel>([NotNull] this Image<TPixel> image, ImageScaling scaling) where TPixel : struct, IPixel<TPixel>
        {
            if (scaling == ImageScaling.Native) return;
            const int threshold = 2000;
            Size size = new Size(image.Width, image.Height);
            if (size.Height > threshold || size.Width > threshold) return; // Skip if the final size is already large enough
            int
                max = size.Height.Max(size.Width),
                scale = threshold / max;
            if (scale == 1) return;
            image.Mutate(x => x.Resize(new ResizeOptions { Size = new Size(size.Width * scale, size.Height * scale), Sampler = new NearestNeighborResampler() }));
        }
    }
}
