using System;
using System.Collections.Generic;

namespace NeuralNetworkDotNet.Extensions.Types
{
    /// <summary>
    /// A custom <see cref="Dictionary{TKey,TValue}"/> that holds disposable values
    /// </summary>
    /// <typeparam name="TKey">The type of the keys in the dictionary</typeparam>
    /// <typeparam name="TValue">The type of the values in the dictionary</typeparam>
    public sealed class DisposableDictionary<TKey, TValue> : Dictionary<TKey, TValue>, IDisposable where TValue : IDisposable
    {
        ~DisposableDictionary() => Dispose();

        /// <inheritdoc/>
        void IDisposable.Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose();
        }

        /// <summary>
        /// Invokes the <see cref="IDisposable.Dispose"/> method without suppressing the finalizer
        /// </summary>
        private void Dispose()
        {
            foreach (var value in Values)
                value.Dispose();
        }
    }
}
