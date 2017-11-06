using System;
using System.Runtime.InteropServices;
using Alea;

namespace NeuralNetworkNET.Cuda.Helpers
{
    public static class CUDAHelper
    {
        /// <summary>
        /// Gets the name of the current CUDA dll in use
        /// </summary>
#if LINUX
        private const string CUDA_DLL_NAME = "libcuda";

#else
        private const string CUDA_DLL_NAME = "nvcuda";
#endif

        public static (ulong Free, ulong Total) GetMemoryInfo()
        {
            //CUResult initResult = cuInit(0);

            CreateContext();

            SizeT
                free = new SizeT(0),
                total = new SizeT(0);
            CUResult result = cuMemGetInfo(ref free, ref total);
            if (result == CUResult.Success) return (free, total);
            throw new InvalidOperationException("Error while retrieving the memory info");
        }

        public static void CreateContext()
        {
            var context = new CUcontext();
            //var error = cuCtxCreate(ref context, 0u, GetHandle(0));

            // ...
            var int1 = context.Pointer.ToInt64();
            context = new CUcontext { Pointer = Gpu.Default.Context.Handle };
            //var int2 = context.Pointer.ToInt64();
            var err2 = cuCtxSetCurrent(context);
        }

        public static CUdevice GetHandle(int ordinal)
        {
            CUdevice udevice = new CUdevice();
            var error = cuDeviceGet(ref udevice, ordinal);
            return udevice;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct CUdevice
        {
            public int Pointer;
        }

        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuCtxCreate_v2")]
        public static extern CUResult cuCtxCreate(ref CUcontext pctx, uint flags, CUdevice dev);

        [DllImport(CUDA_DLL_NAME)]
        private static extern CUResult cuInit(uint Flags);

        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemGetInfo_v2")]
        private static extern CUResult cuMemGetInfo(ref SizeT free, ref SizeT total);

        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuCtxSetCurrent(CUcontext pctx);

        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuDeviceGet(ref CUdevice device, int ordinal);

        [StructLayout(LayoutKind.Sequential)]
        public struct CUcontext
        {
            public IntPtr Pointer;
        }

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

        [StructLayout(LayoutKind.Sequential)]
        public struct SizeT
        {
            private IntPtr value;
            public SizeT(int value)
            {
                this.value = new IntPtr(value);
            }

            public SizeT(uint value)
            {
                this.value = new IntPtr((int)value);
            }

            public SizeT(long value)
            {
                this.value = new IntPtr(value);
            }

            public SizeT(ulong value)
            {
                this.value = new IntPtr((long)value);
            }

            public static implicit operator int(SizeT t)
            {
                return t.value.ToInt32();
            }

            public static implicit operator uint(SizeT t)
            {
                return (uint)((int)t.value);
            }

            public static implicit operator long(SizeT t)
            {
                return t.value.ToInt64();
            }

            public static implicit operator ulong(SizeT t)
            {
                return (ulong)((long)t.value);
            }

            public static implicit operator SizeT(int value)
            {
                return new SizeT(value);
            }

            public static implicit operator SizeT(uint value)
            {
                return new SizeT(value);
            }

            public static implicit operator SizeT(long value)
            {
                return new SizeT(value);
            }

            public static implicit operator SizeT(ulong value)
            {
                return new SizeT(value);
            }

            public static bool operator !=(SizeT val1, SizeT val2)
            {
                return (val1.value != val2.value);
            }

            public static bool operator ==(SizeT val1, SizeT val2)
            {
                return (val1.value == val2.value);
            }
        }
    }
}
