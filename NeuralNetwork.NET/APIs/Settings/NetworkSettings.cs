using System;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Delegates;

namespace NeuralNetworkNET.APIs.Settings
{
    /// <summary>
    /// A static class with some shared settings for the library
    /// </summary>
    public static class NetworkSettings
    {
        private static int _MaximumBatchSize = int.MaxValue;

        /// <summary>
        /// Gets or sets the maximum batch size (used to optimize the memory usage during validation/test processing)
        /// </summary>
        /// <remarks>Adjust this setting to the highest possible value according to the available RAM/VRAM and the size of the dataset. If the validation/test dataset has more
        /// samples than <see cref="MaximumBatchSize"/>, it will be automatically divided into batches so that it won't cause an <see cref="OutOfMemoryException"/> or other problems</remarks>
        public static int MaximumBatchSize
        {
            get => _MaximumBatchSize;
            set => _MaximumBatchSize = value >= 10 ? value : throw new ArgumentOutOfRangeException(nameof(MaximumBatchSize), "The maximum batch size must be at least equal to 10");
        }

        private static AccuracyTester _AccuracyTester = AccuracyTesters.Argmax();

        /// <summary>
        /// Gets or sets the <see cref="Delegates.AccuracyTester"/> instance to use to test a network being trained. The default value is <see cref="AccuracyTesters.Argmax"/>.
        /// </summary>
        [NotNull]
        public static AccuracyTester AccuracyTester
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _AccuracyTester;
            set => _AccuracyTester = value ?? throw new ArgumentNullException(nameof(AccuracyTester), "The input delegate can't be null");
        }
    }
}
