using System;
using Alea;
using JetBrains.Annotations;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Structs;

namespace NeuralNetworkNET.Cuda.Extensions
{
    /// <summary>
    /// An extension class with some additions to the <see cref="Gpu"/> class
    /// </summary>
    internal static class GpuExtensions
    {
        #region Memory copy

        /// <summary>
        /// Allocates a memory area on device memory and copies the contents of the input <see cref="Tensor"/>
        /// </summary>
        /// <param name="gpu">The <see cref="Gpu"/> device to use</param>
        /// <param name="source">The source <see cref="Tensor"/> with the data to copy</param>
        [MustUseReturnValue, NotNull]
        public static unsafe DeviceMemory<float> AllocateDevice([NotNull] this Gpu gpu, in Tensor source)
        {
            DeviceMemory<float> result_gpu = gpu.AllocateDevice<float>(source.Size);
            return CUDAInterop.cuMemcpy(result_gpu.Handle, source.Ptr, new IntPtr(sizeof(float) * source.Size)) == CUDAInterop.cudaError_enum.CUDA_SUCCESS
                ? result_gpu
                : throw new InvalidOperationException("Failed to copy the source data on the target GPU device");
        }

        /// <summary>
        /// Copies the contents of the input <see cref="DeviceMemory{T}"/> instance to the target host memory area
        /// </summary>
        /// <param name="source">The <see cref="DeviceMemory{T}"/> area to read</param>
        /// <param name="destination">The destination <see cref="Tensor"/> to write on</param>
        public static unsafe void CopyTo([NotNull] this DeviceMemory<float> source, in Tensor destination)
        {
            int bytes = sizeof(float) * destination.Size;
            if (bytes != source.Length) throw new ArgumentException("The target tensor doesn't have the same salid as the source GPU memory");
            if (CUDAInterop.cuMemcpy(destination.Ptr, source.Handle, new IntPtr(bytes)) != CUDAInterop.cudaError_enum.CUDA_SUCCESS)
                throw new InvalidOperationException("Failed to copy the source data on the given destination");
        }

        /// <summary>
        /// Copies the contents of the input <see cref="DeviceMemory{T}"/> to a new memory area on the unmanaged heap
        /// </summary>
        /// <param name="source">The source <see cref="DeviceMemory{T}"/> memory to copy</param>
        /// <param name="n">The height of the input memory area</param>
        /// <param name="chw">The width of the input memory area</param>
        /// <param name="result">The resulting matrix</param>
        [MustUseReturnValue]
        public static void CopyToHost([NotNull] this DeviceMemory<float> source, int n, int chw, out Tensor result)
        {
            Tensor.New(n, chw, out result);
            source.CopyTo(result);
        }

        #endregion

        #region Memory copy 2D

        /// <summary>
        /// Allocates a 2D memory area on device memory and copies the contents of the input <see cref="Tensor"/>
        /// </summary>
        /// <param name="gpu">The <see cref="Gpu"/> device to use</param>
        /// <param name="source">The source <see cref="Tensor"/> with the data to copy</param>
        [MustUseReturnValue, NotNull]
        public static unsafe DeviceMemory2D<float> AllocateDevice2D([NotNull] this Gpu gpu, in Tensor source)
        {
            DeviceMemory2D<float> result_gpu = gpu.AllocateDevice<float>(source.Entities, source.Length);
            CUDAInterop.CUDA_MEMCPY2D_st* ptSt = stackalloc CUDAInterop.CUDA_MEMCPY2D_st[1];
            ptSt[0] = new CUDAInterop.CUDA_MEMCPY2D_st
            {
                srcMemoryType = CUDAInterop.CUmemorytype_enum.CU_MEMORYTYPE_HOST,
                srcHost = source.Ptr,
                srcPitch = new IntPtr(sizeof(float) * source.Length),
                dstMemoryType = CUDAInterop.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE,
                dstDevice = result_gpu.Handle,
                dstPitch = result_gpu.Pitch,
                WidthInBytes = new IntPtr(sizeof(float) * source.Length),
                Height = new IntPtr(source.Entities)
            };
            return CUDAInterop.cuMemcpy2D(ptSt) == CUDAInterop.cudaError_enum.CUDA_SUCCESS
                ? result_gpu
                : throw new InvalidOperationException("Failed to copy the source data on the target GPU device");
        }

        /// <summary>
        /// Copies the contents of the input <see cref="DeviceMemory2D{T}"/> instance to the target host memory area
        /// </summary>
        /// <param name="source">The <see cref="DeviceMemory2D{T}"/> area to read</param>
        /// <param name="destination">The destination <see cref="Tensor"/> memory to write on</param>
        public static unsafe void CopyTo([NotNull] this DeviceMemory2D<float> source, in Tensor destination)
        {
            CUDAInterop.CUDA_MEMCPY2D_st* ptSt = stackalloc CUDAInterop.CUDA_MEMCPY2D_st[1];
            ptSt[0] = new CUDAInterop.CUDA_MEMCPY2D_st
            {
                srcMemoryType = CUDAInterop.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE,
                srcDevice = source.Handle,
                srcPitch = source.Pitch,
                dstMemoryType = CUDAInterop.CUmemorytype_enum.CU_MEMORYTYPE_HOST,
                dstHost = destination.Ptr,
                dstPitch = new IntPtr(sizeof(float) * destination.Length),
                WidthInBytes = new IntPtr(sizeof(float) * destination.Length),
                Height = new IntPtr(destination.Entities)
            };
            if (CUDAInterop.cuMemcpy2D(ptSt) != CUDAInterop.cudaError_enum.CUDA_SUCCESS)
                throw new InvalidOperationException("Failed to copy the source data on the given destination");
        }

        /// <summary>
        /// Copies the contents of the input <see cref="DeviceMemory2D{T}"/> to a new memory area on the unmanaged heap
        /// </summary>
        /// <param name="source">The source <see cref="DeviceMemory2D{T}"/> memory to copy</param>
        /// <param name="result">The resulting matrix</param>
        [MustUseReturnValue]
        public static void CopyToHost([NotNull] this DeviceMemory2D<float> source, out Tensor result)
        {
            MemoryLayout.Layout2D layout = source.Layout.To<MemoryLayout, MemoryLayout.Layout2D>();
            Tensor.New(layout.height.ToInt32(), layout.width.ToInt32(), out result);
            source.CopyTo(result);
        }

        #endregion

        /// <summary>
        /// Gets the amount of available GPU memory for a given GPU
        /// </summary>
        /// <param name="gpu">The target <see cref="Gpu"/> to use to retrieve the info</param>
        [Pure]
        public static unsafe (ulong Free, ulong Total) GetFreeMemory([NotNull] this Gpu gpu)
        {
            // Set the context
            CUDAInterop.cudaError_enum result = CUDAInterop.cuCtxSetCurrent(Gpu.Default.Context.Handle);
            if (result != CUDAInterop.cudaError_enum.CUDA_SUCCESS) throw new InvalidOperationException($"Error setting the GPU context: {result}");

            // Get the memory info
            IntPtr* pointers = stackalloc IntPtr[2];
            result = CUDAInterop.cuMemGetInfo(pointers, pointers + 1);
            if (result != 0) throw new InvalidOperationException("Error while retrieving the memory info");
            return ((ulong)pointers[0].ToInt64(), (ulong)pointers[1].ToInt64());
        }
    }
}