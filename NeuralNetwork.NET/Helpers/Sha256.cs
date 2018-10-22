using System;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A static class that can be used to quickly calculate hashes from array of an arbitrary <see langword="struct"/> type
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
        [PublicAPI]
        [Pure, NotNull]
        public static unsafe byte[] Hash<T>([NotNull] T[] array) where T : unmanaged
        {
            int size = Unsafe.SizeOf<T>() * array.Length;
            fixed (T* p = array)
            using (UnmanagedMemoryStream stream = new UnmanagedMemoryStream((byte*)p, size, size, FileAccess.Read))
            using (SHA256 provider = SHA256.Create())
            {
                return provider.ComputeHash(stream);
            }
        }

        /// <summary>
        /// Calculates an aggregate hash for the input <typeparamref name="T"/> arrays
        /// </summary>
        /// <typeparam name="T">The type of items in the input arrays</typeparam>
        /// <param name="arrays">The arrays to process</param>
        [PublicAPI]
        [Pure, NotNull]
        public static unsafe byte[] Hash<T>([NotNull, ItemNotNull] params T[][] arrays) where T : unmanaged
        {
            // Compute the hashes in parallel
            if (arrays.Length == 0) return new byte[0];
            if (arrays.Any(v => v.Length == 0)) throw new ArgumentException("The input array can't contain empty vectors");
            byte[][] hashes = new byte[arrays.Length][];
            Parallel.For(0, arrays.Length, i => hashes[i] = Hash(arrays[i])).AssertCompleted();

            // Merge the computed hashes into a single bytes array
            unchecked
            {
                byte[] result = new byte[HashLength];
                fixed (byte* p = result)
                    for (int i = 0; i < HashLength; i++)
                    {
                        uint hash = 17;
                        for (int j = 0; j < hashes.Length; j++)
                            hash = hash * 31 + hashes[j][i];
                        p[i] = (byte)(hash % byte.MaxValue);
                    }
                return result;
            }
        }
    }
}
