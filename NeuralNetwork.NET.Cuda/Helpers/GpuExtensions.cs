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
        public static ulong GetFreeMemory([NotNull] this Gpu gpu)
        {
            // Set the context
            CUResult result = CUDA_SetContext(Gpu.Default.Context.Handle);
            if (result != CUResult.Success) throw new InvalidOperationException($"Error setting the GPU context: {result}");

            // Get the memory info
            IntPtr
                free = new IntPtr(0),
                total = new IntPtr(0);
            result = CUDA_GetMemInfo(ref free, ref total);
            if (result != CUResult.Success) throw new InvalidOperationException("Error while retrieving the memory info");
            return (ulong)free.ToInt64();
        }

        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemGetInfo_v2")]
        private static extern CUResult CUDA_GetMemInfo(ref IntPtr free, ref IntPtr total);

        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuCtxSetCurrent")]
        public static extern CUResult CUDA_SetContext(IntPtr pctx);

        /// <summary>
        /// Indicates the result of a native call to the CUDA dll
        /// </summary>
        public enum CUResult
        {
            ECCUncorrectable = 0xd6,
            ErrorAlreadyAcquired = 210,
            ErrorAlreadyMapped = 0xd0,
            ErrorArrayIsMapped = 0xcf,
            ErrorContextAlreadyCurrent = 0xca,
            ErrorDeinitialized = 4,
            ErrorFileNotFound = 0x12d,
            ErrorInvalidContext = 0xc9,
            ErrorInvalidDevice = 0x65,
            ErrorInvalidHandle = 400,
            ErrorInvalidImage = 200,
            ErrorInvalidSource = 300,
            ErrorInvalidValue = 1,
            ErrorLaunchFailed = 700,
            ErrorLaunchIncompatibleTexturing = 0x2bf,
            ErrorLaunchOutOfResources = 0x2bd,
            ErrorLaunchTimeout = 0x2be,
            ErrorMapFailed = 0xcd,
            ErrorNoBinaryForGPU = 0xd1,
            ErrorNoDevice = 100,
            ErrorNotFound = 500,
            ErrorNotInitialized = 3,
            ErrorNotMapped = 0xd3,
            ErrorNotReady = 600,
            ErrorOutOfMemory = 2,
            ErrorUnknown = 0x3e7,
            ErrorUnmapFailed = 0xce,
            NotMappedAsArray = 0xd4,
            NotMappedAsPointer = 0xd5,
            PointerIs64Bit = 800,
            SizeIs64Bit = 0x321,
            Success = 0,
            ErrorLaunchTimeOut = 702,
            ErrorPeerAccessNotEnabled = 705,
            ErrorPeerAccessAlreadyEnabled = 704,
            ErrorPrimaryContextActive = 708,
            ErrorContextIsDestroyed = 709,
            ErrorAssert = 710,
            ErrorTooManyPeers = 711,
            ErrorHostMemoryAlreadyInitialized = 712,
            ErrorHostMemoryNotRegistered = 713
        }
    }
}
