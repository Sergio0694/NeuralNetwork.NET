using System;
using Alea;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Structs;

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
        public static DeviceMemory<float> AllocateDevice([NotNull] this Gpu gpu, in Tensor source)
        {
            DeviceMemory<float> result_gpu = gpu.AllocateDevice<float>(source.Size);
            CUDAInterop.cudaError_enum result = CUDAInterop.cuMemcpy(result_gpu.Handle, source.Ptr, new IntPtr(sizeof(float) * source.Size));
            return result == CUDAInterop.cudaError_enum.CUDA_SUCCESS
                ? result_gpu
                : throw new InvalidOperationException($"Failed to copy the source data on the target GPU device, [CUDA ERROR] {result}");
        }

        /// <summary>
        /// Copies the contents of the input <see cref="DeviceMemory{T}"/> instance to the target host memory area
        /// </summary>
        /// <param name="source">The <see cref="DeviceMemory{T}"/> area to read</param>
        /// <param name="destination">The destination <see cref="Tensor"/> to write on</param>
        public static void CopyTo([NotNull] this DeviceMemory<float> source, in Tensor destination)
        {
            if (destination.Size != source.Length) throw new ArgumentException("The target tensor doesn't have the same size as the source GPU memory");
            CUDAInterop.cudaError_enum result = CUDAInterop.cuMemcpy(destination.Ptr, source.Handle, new IntPtr(sizeof(float) * destination.Size));
            if (result != CUDAInterop.cudaError_enum.CUDA_SUCCESS)
                throw new InvalidOperationException($"Failed to copy the source data on the given destination, [CUDA ERROR] {result}");
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

        /// <summary>
        /// Copies the source data into the target <see cref="Tensor"/>, splitting each individual entry into its own row
        /// </summary>
        /// <param name="source">The source memory area with the concatenated data for each entry</param>
        /// <param name="destination">The destination <see cref="Tensor"/> that will store the data</param>
        /// <param name="offset">The column offset for the data for each entry</param>
        /// <param name="length">The number of values to copy for each entry</param>
        public static unsafe void CopyToRows([NotNull] this DeviceMemory<float> source, in Tensor destination, int offset, int length)
        {
            if (source.Length / length != destination.Entities) throw new ArgumentOutOfRangeException(nameof(length), "The input length doesn't match the given arguments");
            if (destination.Length - offset > length) throw new ArgumentOutOfRangeException(nameof(offset), "The input offset isn't valid");
            CUDAInterop.cudaError_enum result = CUDAInterop.cudaError_enum.CUDA_SUCCESS;
            for (int i = 0; i < destination.Entities; i++)
                result |= CUDAInterop.cuMemcpy(new IntPtr((float*)destination + offset), source.Handle + i * destination.Length, new IntPtr(sizeof(float) * length));
            if (result != CUDAInterop.cudaError_enum.CUDA_SUCCESS)
                throw new InvalidOperationException($"Failed to copy the source data on the given destination, [CUDA ERROR] {result}");
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
            if (result != CUDAInterop.cudaError_enum.CUDA_SUCCESS) throw new InvalidOperationException($"Error setting the GPU context, [CUDA ERROR] {result}");

            // Get the memory info
            IntPtr* pointers = stackalloc IntPtr[2];
            result = CUDAInterop.cuMemGetInfo(pointers, pointers + 1);
            if (result != CUDAInterop.cudaError_enum.CUDA_SUCCESS) throw new InvalidOperationException($"Error while retrieving the memory info, [CUDA ERROR] {result}");
            return ((ulong)pointers[0].ToInt64(), (ulong)pointers[1].ToInt64());
        }
    }
}