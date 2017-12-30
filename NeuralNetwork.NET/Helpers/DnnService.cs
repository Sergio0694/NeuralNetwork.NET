using System;
using System.Threading;
using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.DependencyInjections;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A static class that handles a shared, disposable instance of the <see cref="Dnn"/> class
    /// </summary>
    internal static class DnnService
    {
        #region Fields and tools

        // Static weak reference to avoid memory leaks
        [NotNull]
        private static readonly WeakReference<Dnn> DnnReference = new WeakReference<Dnn>(null);

        // The id of the current thread
        private static int _ThreadId = Thread.CurrentThread.ManagedThreadId;

        /// <summary>
        /// Synchronizes the context of the <see cref="Gpu"/> instance in use, if needed
        /// </summary>
        private static void SynchronizeDnnContext()
        {
            lock (DnnReference)
            {
                int id = Thread.CurrentThread.ManagedThreadId;
                if (DnnReference.TryGetTarget(out Dnn dnn) && dnn != null && _ThreadId != id)
                {
                    _ThreadId = id;
                    dnn.Gpu.Context.SetCurrent();
                }
            }
        }

        #endregion

        /// <summary>
        /// Gets a the shared <see cref="Dnn"/> instance in use
        /// </summary>
        [NotNull]
        public static Dnn Instance
        {
            [Pure]
            get
            {
                lock (DnnReference)
                {
                    if (DnnReference.TryGetTarget(out Dnn dnn) && dnn != null) return dnn;
                    dnn = Dnn.Get(Gpu.Default);
                    DnnReference.SetTarget(dnn);
                    LibraryRuntimeHelper.SynchronizeContext = SynchronizeDnnContext;
                    return dnn;
                }
            }
        }
    }
}
