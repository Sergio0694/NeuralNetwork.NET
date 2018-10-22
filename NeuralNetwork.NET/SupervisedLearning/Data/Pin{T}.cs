using System.Runtime.CompilerServices;

namespace NeuralNetworkNET.SupervisedLearning.Data
{
    /// <summary>
    /// A simple <see langword="struct"/> that holds a reference to a pinned target.
    /// It can be used to replace <see cref="System.Span{T}"/> to quickly pass pointers around in non stack-only methods.
    /// </summary>
    /// <typeparam name="T">The type of the target referenced by the exposed pointer</typeparam>
    internal readonly unsafe struct Pin<T> where T : unmanaged
    {
        /// <summary>
        /// The pinned pointer
        /// </summary>
        public readonly void* Ptr;

        /// <summary>
        /// Gets a managed reference for the pointer target
        /// </summary>
        public ref T Ref
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Unsafe.AsRef<T>(Ptr);
        }

        /// <summary>
        /// Gets a managed reference of the input type, using an unsafe cast
        /// </summary>
        /// <typeparam name="TTo">The target reference type to return</typeparam>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref TTo As<TTo>() => ref Unsafe.AsRef<TTo>(Ptr);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Pin(void* p) => Ptr = p;

        /// <summary>
        /// Creates a new instance from the input reference
        /// </summary>
        /// <param name="value">The target reference</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Pin<T> From(ref T value) => new Pin<T>(Unsafe.AsPointer(ref value));
    }
}
