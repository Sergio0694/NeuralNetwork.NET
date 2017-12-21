using System;
using System.IO;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Extensions
{
    /// <summary>
    /// A static class with some <see cref="Stream"/> extensions to load/write contents
    /// </summary>
    internal static class StreamExtensions
    {
        /// <summary>
        /// Writes the input <see cref="struct"/> to the target <see cref="Stream"/> instance
        /// </summary>
        /// <typeparam name="T">The <see cref="struct"/> type to serialize</typeparam>
        /// <param name="stream">The target <see cref="Stream"/> to use to write the data</param>
        /// <param name="value">A reference to the <see cref="struct"/> to write to the <see cref="Stream"/> instance</param>
        public static unsafe void Write<T>([NotNull] this Stream stream, ref T value) where T : struct
        {
            byte[] bytes = new byte[Unsafe.SizeOf<T>()];
            fixed (void* p = bytes) Unsafe.Copy(p, ref value);
            stream.Write(bytes, 0, bytes.Length);
        }

        /// <summary>
        /// Reads a value of the given <see cref="struct"/> type from the input <see cref="Stream"/> instance
        /// </summary>
        /// <typeparam name="T">The <see cref="struct"/> type to read and return</typeparam>
        /// <param name="stream">The source <see cref="Stream"/> instance to use to read the data</param>
        public static unsafe T Read<T>([NotNull] this Stream stream) where T : struct
        {
            byte[] bytes = new byte[Unsafe.SizeOf<T>()];
            stream.Read(bytes, 0, bytes.Length);
            T value = new T();
            fixed (void* p = bytes) Unsafe.Copy(ref value, p);
            return value;
        }

        /// <summary>
        /// Writes a 32-bits int to the target <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The target <see cref="Stream"/></param>
        /// <param name="n">The value to write</param>
        public static void Write([NotNull] this Stream stream, int n) => stream.Write(BitConverter.GetBytes(n), 0, sizeof(int));

        /// <summary>
        /// Reads a 32-bits int from the target <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The target <see cref="Stream"/></param>
        public static int ReadInt32([NotNull] this Stream stream)
        {
            byte[] bytes = new byte[sizeof(int)];
            stream.Read(bytes, 0, sizeof(int));
            return BitConverter.ToInt32(bytes, 0);
        }

        /// <summary>
        /// Writes a <see cref="float"/> matrix to the target <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The target <see cref="Stream"/></param>
        /// <param name="m">The source matrix</param>
        public static unsafe void Write([NotNull] this Stream stream, [NotNull] float[,] m)
        {
            const int blockSize = 1024;
            byte[] temp = new byte[sizeof(float) * blockSize];
            int remaining = m.Length;
            while (remaining > 0)
            {
                int chunkSize = blockSize >= remaining ? remaining : blockSize;
                fixed (float* pm = m)
                fixed (byte* ptemp = temp)
                    Buffer.MemoryCopy(pm + (m.Length - remaining), ptemp, temp.Length, sizeof(float) * chunkSize);
                remaining -= chunkSize;
                stream.Write(temp, 0, sizeof(float) * chunkSize);
            }
        }

        /// <summary>
        /// Reads a <see cref="float"/> matrix from the target <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The source <see cref="Stream"/></param>
        /// <param name="height">The matrix height</param>
        /// <param name="width">The matrix width</param>
        public static float[,] ReadFloatArray([NotNull] this Stream stream, int height, int width)
        {
            float[,] result = new float[height, width];
            const int blockSize = 1024;
            byte[] temp = new byte[sizeof(float) * blockSize];
            int total = height * width, remaining = total;
            while (remaining > 0)
            {
                int chunkSize = blockSize >= remaining ? remaining : blockSize;
                stream.Read(temp, 0, sizeof(float) * chunkSize);
                Buffer.BlockCopy(temp, 0, result, sizeof(float) * (total - remaining), sizeof(float) * chunkSize);
                remaining -= chunkSize;
            }
            return result;
        }

        /// <summary>
        /// Writes a <see cref="float"/> vector to the target <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The target <see cref="Stream"/></param>
        /// <param name="v">The source vector</param>
        public static unsafe void Write([NotNull] this Stream stream, [NotNull] float[] v)
        {
            const int blockSize = 1024;
            byte[] temp = new byte[sizeof(float) * blockSize];
            int remaining = v.Length;
            while (remaining > 0)
            {
                int chunkSize = blockSize >= remaining ? remaining : blockSize;
                fixed (float* pv = v)
                fixed (byte* ptemp = temp)
                    Buffer.MemoryCopy(pv + (v.Length - remaining), ptemp, temp.Length, sizeof(float) * chunkSize);
                remaining -= chunkSize;
                stream.Write(temp, 0, sizeof(float) * chunkSize);
            }
        }

        /// <summary>
        /// Reads a <see cref="float"/> vector from the target <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The source <see cref="Stream"/></param>
        /// <param name="length">The vector length</param>
        public static float[] ReadFloatArray([NotNull] this Stream stream, int length)
        {
            float[] result = new float[length];
            const int blockSize = 1024;
            byte[] temp = new byte[sizeof(float) * blockSize];
            int remaining = length;
            while (remaining > 0)
            {
                int chunkSize = blockSize >= remaining ? remaining : blockSize;
                stream.Read(temp, 0, sizeof(float) * chunkSize);
                Buffer.BlockCopy(temp, 0, result, sizeof(float) * (length - remaining), sizeof(float) * chunkSize);
                remaining -= chunkSize;
            }
            return result;
        }
    }
}
