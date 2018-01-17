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
    }
}
