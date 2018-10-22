using System.IO;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Extensions
{
    /// <summary>
    /// A static class with some <see cref="Stream"/> extensions to load/write contents
    /// </summary>
    internal static class StreamExtensions
    {
        /// <summary>
        /// Writes the input <see langword="struct"/> to the target <see cref="Stream"/> instance
        /// </summary>
        /// <typeparam name="T">The <see langword="struct"/> type to serialize</typeparam>
        /// <param name="stream">The target <see cref="Stream"/> to use to write the data</param>
        /// <param name="value">The <see langword="struct"/> to write to the <see cref="Stream"/> instance</param>
        public static unsafe void Write<T>([NotNull] this Stream stream, T value) where T : unmanaged
        {
            byte[] bytes = new byte[Unsafe.SizeOf<T>()];
            fixed (void* p = bytes) Unsafe.Copy(p, ref value);
            stream.Write(bytes, 0, bytes.Length);
        }

        /// <summary>
        /// Reads a value of the given <see langword="struct"/> type from the input <see cref="Stream"/> instance
        /// </summary>
        /// <typeparam name="T">The <see langword="struct"/> type to read and return</typeparam>
        /// <param name="stream">The source <see cref="Stream"/> instance to use to read the data</param>
        /// <param name="value">The resulting <typeparamref name="T"/> value that is read from the <see cref="Stream"/></param>
        [MustUseReturnValue]
        public static unsafe bool TryRead<T>([NotNull] this Stream stream, out T value) where T : unmanaged
        {
            byte[] bytes = new byte[Unsafe.SizeOf<T>()];
            value = default;
            if (stream.Read(bytes, 0, bytes.Length) == 0) return false;            
            fixed (void* p = bytes) Unsafe.Copy(ref value, p);
            return true;
        }

        /// <summary>
        /// Writes a shuffled <see cref="float"/> vector to the target <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The target <see cref="Stream"/></param>
        /// <param name="v">The source vector to shuffle and write to the <see cref="Stream"/> instance</param>
        [CollectionAccess(CollectionAccessType.Read)]
        public static void WriteShuffled([NotNull] this Stream stream, [NotNull] float[] v)
        {
            byte[] temp = new byte[v.Length * sizeof(float)];
            unsafe void Kernel(int i)
            {
                fixed (void* pv0 = v)
                fixed (byte* pt0 = temp)
                {
                    byte* 
                        pv = (byte*)pv0 + i,
                        pt = pt0 + i * v.Length;
                    for (int j = 0; j < v.Length; j++)
                        pt[j] = pv[j * 4];
                }
            }
            Parallel.For(0, sizeof(float), Kernel).AssertCompleted();
            stream.Write(temp, 0, temp.Length);            
        }

        /// <summary>
        /// Reads a shuffled <see cref="float"/> vector from the target <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The source <see cref="Stream"/></param>
        /// <param name="n">The vector length</param>
        [MustUseReturnValue, NotNull]
        public static float[] ReadUnshuffled([NotNull] this Stream stream, int n)
        {
            // Read the shuffled bytes
            float[] v = new float[n];
            byte[] temp = new byte[n * sizeof(float)];
            stream.Read(temp, 0, temp.Length);

            // Unshuffle in parallel
            unsafe void Kernel(int i)
            {
                fixed (void* pv0 = v)
                fixed (byte* pt0 = temp)
                {
                    byte* 
                        pv = (byte*)pv0 + i,
                        pt = pt0 + i * n;
                    for (int j = 0; j < n; j++)
                        pv[j * 4] = pt[j];
                }
            }
            Parallel.For(0, sizeof(float), Kernel).AssertCompleted();
            return v;
        }
    }
}
