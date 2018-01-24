using System;
using System.Collections.Generic;
using NeuralNetworkNET.APIs.Structs;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A simple disposable map that stores <see cref="Tensor"/> instances while training or using a network
    /// </summary>
    internal sealed class TensorMap<T> : Dictionary<T, Tensor>, IDisposable where T : class
    {
        ~TensorMap() => Dispose();

        /// <inheritdoc/>
        void IDisposable.Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose();
        }

        // Frees the allocated tensors
        private void Dispose()
        {
            foreach (Tensor tensor in Values)
                tensor.Free();
        }

        #if DEBUG

        /// <summary>
        /// Gets or sets a unique value for the input key
        /// </summary>
        /// <param name="key">The key to use to read/data from the map</param>
        /// <exception cref="InvalidOperationException">The key already exists in the map</exception>
        public new Tensor this[T key]
        {
            [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
            get => base[key];
            [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
            set
            {
                if (ContainsKey(key)) throw new InvalidOperationException("The map already contains a value for the current key");
                base[key] = value;
            }
        }

        #endif
    }
}
