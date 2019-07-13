using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkDotNet.Helpers;

namespace System
{
    /// <summary>
    /// An helper <see langword="class"/> with methods to process fixed-size matrices
    /// </summary>
    public static class SpanExtensions
    {
        /// <summary>
        /// Returns an hash code for the contents of the input <see cref="Span{T}"/>
        /// </summary>
        /// <typeparam name="T">The type of each value in the input <see cref="Span{T}"/></typeparam>
        /// <param name="span">The input <see cref="Span{T}"/> to read</param>
        [Pure]
        public static int GetContentHashCode<T>(this Span<T> span) where T : unmanaged
        {
            if (span.Length == 0) return -1;

            var hash = span[0].GetHashCode();
            unchecked
            {
                for (var i = 1; i < span.Length; i++)
                    hash = (hash * 397) ^ i.GetHashCode();
                return hash;
            }
        }

        /// <summary>
        /// Fills the target <see cref="Span{T}"/> with the input values provider
        /// </summary>
        /// <param name="span">The <see cref="Span{T}"/> to fill up</param>
        /// <param name="provider">The values provider to use</param>
        public static void Fill<T>(this Span<T> span, [NotNull] Func<T> provider) where T : unmanaged
        {
            var l = span.Length;
            ref var r = ref span.GetPinnableReference();
            for (var i = 0; i < l; i++)
                Unsafe.Add(ref r, i) = provider();
        }

        /// <summary>
        /// Checks whether or not the two input <see cref="Span{T}"/> instances have the same content
        /// </summary>
        /// <param name="a">The first <see cref="Span{T}"/> to check</param>
        /// <param name="b">The second <see cref="Span{T}"/> to check</param>
        /// <param name="threshold">The threshold to use for the comparisons</param>
        [Pure]
        public static bool ContentEquals(this Span<float> a, Span<float> b, float threshold = 0.0001f)
        {
            Guard.IsFalse(threshold <= 0, nameof(threshold), "The threshold must be a positive number");

            if (a.Length != b.Length) return false;
            if (a.Length == 0) return true;

            var l = a.Length;
            ref var ra = ref a.GetPinnableReference();
            ref var rb = ref b.GetPinnableReference();

            for (var i = 0; i < l; i++)
                if (Math.Abs(Unsafe.Add(ref ra, i) - Unsafe.Add(ref rb, i)) > threshold)
                    return false;

            return true;
        }

        /// <summary>
        /// Returns the index of the maximum value in the input <see cref="Span{T}"/>
        /// </summary>
        /// <param name="span">The source <see cref="Span{T}"/> instance</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public static int Argmax(this Span<float> span)
        {
            if (span.Length < 2) return default;

            int index = 0, l = span.Length;
            var max = float.MinValue;
            ref var r = ref span.GetPinnableReference();

            for (var j = 0; j < l; j++)
            {
                var value = Unsafe.Add(ref r, j);
                if (value > max)
                {
                    max = value;
                    index = j;
                }
            }

            return index;
        }

        /// <summary>
        /// Returns whether or not the input <see cref="Span{T}"/> contains at least one <see cref="float.NaN"/> value
        /// </summary>
        /// <param name="span">The source <see cref="Span{T}"/> instance</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public static bool HasNaN(this Span<float> span)
        {
            var l = span.Length;
            ref var r = ref span.GetPinnableReference();

            for (var w = 0; w < l; w++)
                if (float.IsNaN(Unsafe.Add(ref r, w)))
                    return true;

            return false;
        }
    }
}
