using System;
using System.IO;
using System.Linq;
using System.Threading;
using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Services;

namespace NeuralNetworkNET.cuDNN
{
    /// <summary>
    /// A static class that handles a shared, disposable instance of the <see cref="Dnn"/> class
    /// </summary>
    internal static class CuDnnService
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
                if (DnnReference.TryGetTarget(out Dnn dnn) && _ThreadId != id)
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
                    if (DnnReference.TryGetTarget(out Dnn dnn)) return dnn;
                    dnn = Dnn.Get(Gpu.Default);
                    DnnReference.SetTarget(dnn);
                    SharedEventsService.TrainingStarting.Add(SynchronizeDnnContext);
                    return dnn;
                }
            }
        }

        #region Availability check

        /// <summary>
        /// Gets whether or not the cuDNN support is available on the current system
        /// </summary>
        public static bool IsAvailable
        {
            get
            {
                try
                {
                    // Calling this directly could cause a crash in the <Module> loader due to the missing .dll files
                    return CuDnnSupportHelper.IsGpuAccelerationSupported();
                }
                catch (Exception e) when (e is FileNotFoundException || e is TypeInitializationException)
                {
                    // Missing .dll file
                    return false;
                }
            }
        }

        /// <summary>
        /// A private class that is used to create a new standalone type that contains the actual test method (decoupling is needed to &lt;Module&gt; loading crashes)
        /// </summary>
        private static class CuDnnSupportHelper
        {
            /// <summary>
            /// Checks whether or not the Cuda features are currently supported
            /// </summary>
            public static bool IsGpuAccelerationSupported()
            {
                try
                {
                    // CUDA test
                    Gpu gpu = Gpu.Default;
                    if (gpu == null) return false;
                    if (!Dnn.IsAvailable) return false; // cuDNN
                    using (DeviceMemory<float> sample_gpu = gpu.AllocateDevice<float>(1024))
                    {
                        deviceptr<float> ptr = sample_gpu.Ptr;
                        void Kernel(int i) => ptr[i] = i;
                        Alea.Parallel.GpuExtension.For(gpu, 0, 1024, Kernel); // JIT test
                        float[] sample = Gpu.CopyToHost(sample_gpu);
                        return Enumerable.Range(0, 1024).Select<int, float>(i => i).ToArray().ContentEquals(sample);
                    }
                }
                catch
                {
                    // Missing .dll or other errors
                    return false;
                }
            }
        }

        #endregion
    }
}
