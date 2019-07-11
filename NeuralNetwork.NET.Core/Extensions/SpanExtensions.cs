using System;
using JetBrains.Annotations;

namespace NeuralNetwork.NET.Core.Extensions
{
    /// <summary>
    /// An helper class with methods to process fixed-size matrices
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
            int hash = 17;
            unchecked
            {
                for (int i = 0; i < span.Length; i++)
                    hash = hash * 23 + span[i].GetHashCode();
                return hash;
            }
        }
    }
}
