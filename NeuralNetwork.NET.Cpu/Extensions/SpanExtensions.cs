using System.Runtime.CompilerServices;
using JetBrains.Annotations;

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
    }
}
