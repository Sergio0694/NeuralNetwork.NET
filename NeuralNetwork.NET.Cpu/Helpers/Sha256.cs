using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using JetBrains.Annotations;

namespace NeuralNetworkDotNet.Helpers
{
    /// <summary>
    /// A <see langword="class"/> that can be used to quickly calculate hashes from arrays of arbitrary types
    /// </summary>
    public sealed class Sha256
    {
        /// <summary>
        /// The hash data for the current instance
        /// </summary>
        [NotNull]
        private readonly byte[] Bytes;

        private Sha256([NotNull] byte[] bytes) => Bytes = bytes;

        /// <summary>
        /// Calculates an hash for the input <see cref="Span{T}"/> instance
        /// </summary>
        /// <typeparam name="T">The type of items in the input <see cref="Span{T}"/></typeparam>
        /// <param name="span">The input data to process</param>
        [Pure, NotNull]
        public static unsafe Sha256 Hash<T>(Span<T> span) where T : unmanaged
        {
            var size = Unsafe.SizeOf<T>() * span.Length;
            fixed (T* p = span)
            {
                using (var stream = new UnmanagedMemoryStream((byte*)p, size, size, FileAccess.Read))
                using (var provider = SHA256.Create())
                {
                    var hash = provider.ComputeHash(stream);
                    return new Sha256(hash);
                }
            }
        }

        // The SHA256 hash bytes length
        private const int HashLength = 32;

        /// <summary>
        /// Combines the current <see cref="Sha256"/> instance with a new <see cref="Span{T}"/>
        /// </summary>
        /// <typeparam name="T">The type of items in the input arrays</typeparam>
        /// <param name="span">The input data to process</param>
        [Pure, NotNull]
        public Sha256 And<T>(Span<T> span) where T : unmanaged
        {
            var bytes = Hash(span).Bytes;

            // Merge the computed hashes into a single bytes array
            unchecked
            {
                ref var rb1 = ref Bytes.AsSpan().GetPinnableReference();
                ref var rb2 = ref bytes.AsSpan().GetPinnableReference();

                for (var i = 0; i < HashLength; i++)
                {
                    uint hash = Unsafe.Add(ref rb1, i);
                    hash = (hash * 397) ^ Unsafe.Add(ref rb2, i);
                    Unsafe.Add(ref rb2, i) = (byte)(hash % byte.MaxValue);
                }

                return new Sha256(bytes);
            }
        }

        /// <inheritdoc/>
        public override string ToString() => Convert.ToBase64String(Bytes);
    }
}
