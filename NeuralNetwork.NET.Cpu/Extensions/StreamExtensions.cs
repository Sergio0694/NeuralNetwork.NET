using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading;
using JetBrains.Annotations;
using NeuralNetworkDotNet.Helpers;

namespace NeuralNetworkDotNet.Extensions
{
    /// <summary>
    /// A <see langword="class"/> with some <see cref="Stream"/> extensions to load/write data
    /// </summary>
    internal static class StreamExtensions
    {
        /// <summary>
        /// The maximum size of values to serialize
        /// </summary>
        private const int BufferSize = 128;

        /// <summary>
        /// The <see cref="ThreadLocal{T}"/> instance that provides reusable buffers
        /// </summary>
        [NotNull]
        private static readonly ThreadLocal<byte[]> Buffer = new ThreadLocal<byte[]>(() => new byte[BufferSize]);

        /// <summary>
        /// Writes the input <see langword="struct"/> to the target <see cref="Stream"/> instance
        /// </summary>
        /// <typeparam name="T">The <see langword="struct"/> type to serialize</typeparam>
        /// <param name="stream">The target <see cref="Stream"/> to use to write the data</param>
        /// <param name="value">The <see langword="struct"/> to write to the <see cref="Stream"/> instance</param>
        public static void Write<T>([NotNull] this Stream stream, T value) where T : unmanaged
        {
            var l = Unsafe.SizeOf<T>();

            Guard.IsFalse(l > BufferSize, "The input struct type is too large");

            ref var rx = ref Unsafe.As<T, byte>(ref value);
            ref var ry = ref Buffer.Value[0];

            for (var i = 0; i < l; i++)
                Unsafe.Add(ref ry, i) = Unsafe.Add(ref rx, l - 1 - i);

            stream.Write(Buffer.Value, 0, l);
        }

        /// <summary>
        /// Reads a value of the given <see langword="struct"/> type from the input <see cref="Stream"/> instance
        /// </summary>
        /// <typeparam name="T">The <see langword="struct"/> type to read and return</typeparam>
        /// <param name="stream">The source <see cref="Stream"/> instance to use to read the data</param>
        /// <param name="value">The resulting <typeparamref name="T"/> value that is read from the <see cref="Stream"/></param>
        [MustUseReturnValue]
        public static bool TryRead<T>([NotNull] this Stream stream, out T value) where T : unmanaged
        {
            var l = Unsafe.SizeOf<T>();
            value = default;

            Guard.IsFalse(l > BufferSize, "The input struct type is too large");

            if (stream.Read(Buffer.Value, 0, l) != l) return false;

            ref var rx = ref Unsafe.As<T, byte>(ref value);
            ref var ry = ref Buffer.Value.AsSpan().GetPinnableReference();

            for (var i = 0; i < l; i++)
                Unsafe.Add(ref rx, i) = Unsafe.Add(ref ry, l - 1 - i);

            return true;
        }

        /// <summary>
        /// Writes a given <see cref="Span{T}"/> to the target <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The target <see cref="Stream"/></param>
        /// <param name="span">The source <see cref="Span{T}"/> to write to the <see cref="Stream"/> instance</param>
        [CollectionAccess(CollectionAccessType.Read)]
        public static void Write<T>([NotNull] this Stream stream, Span<T> span) where T : unmanaged
        {
            var l = Unsafe.SizeOf<T>();
            var buffer = new byte[span.Length];

            ref var rx = ref Unsafe.As<T, byte>(ref span.GetPinnableReference());
            ref var ry = ref buffer[0];

            for (var b = 0; b < l; b++, rx = ref Unsafe.Add(ref rx, 1))
            {
                for (var i = 0; i < span.Length; i++)
                {
                    Unsafe.Add(ref ry, i) = Unsafe.Add(ref rx, i * l);
                }

                stream.Write(buffer, 0, buffer.Length);
            }
        }

        /// <summary>
        /// Reads a shuffled <see cref="float"/> vector from the target <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The source <see cref="Stream"/></param>
        /// <param name="n">The vector length</param>
        [MustUseReturnValue, CanBeNull]
        public static T[] TryRead<T>([NotNull] this Stream stream, int n) where T : unmanaged
        {
            var l = Unsafe.SizeOf<T>();
            var temp = new byte[n * l];
            if (stream.Read(temp, 0, temp.Length) != temp.Length) return null;

            var data = new T[n];

            ref var rx = ref temp[0];
            ref var ry = ref Unsafe.As<T, byte>(ref data[0]);

            for (var b = 0; b < l; b++, ry = ref Unsafe.Add(ref ry, 1))
            {
                for (var i = 0; i < n; i++)
                {
                    Unsafe.Add(ref ry, i * l) = Unsafe.Add(ref rx, i);
                }
            }

            return data;
        }
    }
}
