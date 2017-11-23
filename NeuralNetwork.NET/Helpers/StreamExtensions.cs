using System;
using System.IO;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Helpers
{
    internal static class StreamExtensions
    {
        public static void Write([NotNull] this Stream stream, int n) => stream.Write(BitConverter.GetBytes(n), 0, sizeof(int));

        public static int ReadInt32([NotNull] this Stream stream)
        {
            byte[] bytes = new byte[sizeof(int)];
            stream.Read(bytes, 0, sizeof(int));
            return BitConverter.ToInt32(bytes, 0);
        }

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
                stream.Write(temp, 0, chunkSize);
            }
        }

        public static float[,] ReadFloatArray([NotNull] this Stream stream, int height, int width)
        {
            float[,] result = new float[height, width];
            const int blockSize = 1024;
            byte[] temp = new byte[sizeof(float) * blockSize];
            int total = height * width, remaining = total;
            while (remaining > 0)
            {
                int chunkSize = blockSize >= remaining ? remaining : blockSize;
                stream.Read(temp, 0, chunkSize);
                Buffer.BlockCopy(temp, 0, result, total - remaining, chunkSize);
                remaining -= chunkSize;
            }
            return result;
        }

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
                stream.Write(temp, 0, chunkSize);
            }
        }

        public static float[] ReadFloatArray([NotNull] this Stream stream, int length)
        {
            float[] result = new float[length];
            const int blockSize = 1024;
            byte[] temp = new byte[sizeof(float) * blockSize];
            int remaining = length;
            while (remaining > 0)
            {
                int chunkSize = blockSize >= remaining ? remaining : blockSize;
                stream.Read(temp, 0, chunkSize);
                Buffer.BlockCopy(temp, 0, result, length - remaining, chunkSize);
                remaining -= chunkSize;
            }
            return result;
        }
    }
}
