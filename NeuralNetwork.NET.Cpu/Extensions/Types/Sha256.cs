using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkDotNet.Helpers;

namespace System.Security.Cryptography
{
    /// <summary>
    /// A <see langword="class"/> that can be used to quickly calculate hashes from arrays of arbitrary types
    /// </summary>
    public static class Sha256
    {
        // The SHA256 hash bytes length
        private const int HashLength = 32;

        /// <summary>
        /// Calculates an hash for the input <typeparamref name="T"/> array
        /// </summary>
        /// <typeparam name="T">The type of items in the input array</typeparam>
        /// <param name="array">The input array to process</param>
        [Pure, NotNull]
        public static unsafe byte[] Hash<T>([NotNull] T[] array) where T : unmanaged
        {
            var size = Unsafe.SizeOf<T>() * array.Length;
            fixed (T* p = array)
            {
                using (var stream = new UnmanagedMemoryStream((byte*)p, size, size, FileAccess.Read))
                using (var provider = SHA256.Create())
                {
                    return provider.ComputeHash(stream);
                }
            }
        }

        /// <summary>
        /// Calculates an aggregate hash for the input <typeparamref name="T"/> arrays
        /// </summary>
        /// <typeparam name="T">The type of items in the input arrays</typeparam>
        /// <param name="arrays">The arrays to process</param>
        [Pure, NotNull]
        public static unsafe byte[] Hash<T>([NotNull, ItemNotNull] params T[][] arrays) where T : unmanaged
        {
            if (arrays.Length == 0) return new byte[0];

            Guard.IsFalse(arrays.Any(v => v.Length == 0), nameof(arrays), "The input array can't contain empty vectors");

            // Compute the hashes in parallel
            var hashes = new byte[arrays.Length][];
            Parallel.For(0, arrays.Length, i => hashes[i] = Hash(arrays[i]));

            // Merge the computed hashes into a single bytes array
            unchecked
            {
                var result = new byte[HashLength];
                fixed (byte* p = result)
                {
                    for (var i = 0; i < HashLength; i++)
                    {
                        uint hash = 17;
                        for (var j = 0; j < hashes.Length; j++)
                            hash = hash * 31 + hashes[j][i];
                        p[i] = (byte)(hash % byte.MaxValue);
                    }
                }

                return result;
            }
        }
    }
}
