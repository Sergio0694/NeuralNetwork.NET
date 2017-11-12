using System;
using System.Runtime.InteropServices;
using Alea;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Cuda.Helpers
{
    /// <summary>
    /// An extension class with some additions to the <see cref="Gpu"/> class
    /// </summary>
    public static class GpuExtensions
    {
        /// <summary>
        /// Gets the name of the current CUDA dll in use
        /// </summary>
#if LINUX
        private const string CUDA_DLL_NAME = "libcuda";

#else
        private const string CUDA_DLL_NAME = "nvcuda";
#endif

        /// <summary>
        /// Gets the amount of available GPU memory for a given GPU
        /// </summary>
        /// <param name="gpu">The target <see cref="Gpu"/> to use to retrieve the info</param>
        [PublicAPI]
        public static ulong GetFreeMemory([NotNull] this Gpu gpu)
        {
            // Set the context
            int result = CUDA_SetContext(Gpu.Default.Context.Handle);
            if (result != 0) throw new InvalidOperationException($"Error setting the GPU context: {result}");

            // Get the memory info
            IntPtr
                free = new IntPtr(0),
                total = new IntPtr(0);
            result = CUDA_GetMemInfo(ref free, ref total);
            if (result != 0) throw new InvalidOperationException("Error while retrieving the memory info");
            return (ulong)free.ToInt64();
        }

        // Gets the info on the amount of free and total GPU memory available
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemGetInfo_v2")]
        private static extern int CUDA_GetMemInfo(ref IntPtr free, ref IntPtr total);

        // Sets the GPU context for subsequent calls to the CUDA library
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuCtxSetCurrent")]
        public static extern int CUDA_SetContext(IntPtr pctx);
    }
}
