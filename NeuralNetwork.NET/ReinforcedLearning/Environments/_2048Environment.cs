using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.ReinforcedLearning.Environments
{
    /// <summary>
    /// An environment for the 2048 game
    /// </summary>
    public sealed class _2048Environment : IEnvironment
    {
        /// <inheritdoc/>
        public int Size { get; } = 16;

        /// <inheritdoc/>
        public int Actions { get; } = 4;

        /// <inheritdoc/>
        public int Reward { get; }

        /// <inheritdoc/>
        public int Timestep { get; }

        /// <inheritdoc/>
        public bool CanExecute
        {
            get
            {
                ref int r = ref Data[0];
                for (int i = 0; i < 16; i++)
                {
                    int value = Unsafe.Add(ref r, i);
                    if (value == 2048) return false;
                    if (value == 0) return true;
                }

                return false;
            }
        }

        /// <summary>
        /// A singleton <see cref="ArrayPool{T}"/> instance to quickly create new instances of the environment
        /// </summary>
        [NotNull]
        private static readonly ArrayPool<int> Allocator = ArrayPool<int>.Create(32, 1000);

        /// <summary>
        /// The array representing the environment current state
        /// </summary>
        [NotNull]
        private readonly int[] Data = Allocator.Rent(32);

        /// <summary>
        /// Creates a new, empty <see cref="TicTacToeEnvironment"/> instance
        /// </summary>
        public _2048Environment()
        {
            Data[ThreadSafeRandom.NextInt(max: 4) * 4 + ThreadSafeRandom.NextInt(max: 4)] = ThreadSafeRandom.NextInt(1, 3) * 2;
            Span<int> free = GetFreePositions();
            int next = ThreadSafeRandom.NextInt(max: free.Length);
            Data[free[next]] = ThreadSafeRandom.NextInt(1, 3) * 2;

        }

        private _2048Environment(int reward, int timestep)
        {
            Reward = reward;
            Timestep = timestep;
        }

        /// <inheritdoc/>
        public IEnvironment Execute(int action)
        {
            // TODO
            return null;
        }

        /// <inheritdoc/>
        public void Serialize(Span<float> span)
        {
            ref int rdata = ref Data[0];
            ref float rspan = ref span.GetPinnableReference();

            for (int i = 0; i < 16; i++)
            {
                Unsafe.Add(ref rspan, i) = Unsafe.Add(ref rdata, i) / 2048f;
            }
        }

        /// <inheritdoc/>
        public IEnvironment Clone()
        {
            var instance = new _2048Environment(Reward, Timestep);
            Data.AsSpan(0, Size).CopyTo(instance.Data.AsSpan(0, Size));
            return instance;
        }

        /// <inheritdoc/>
        public bool Equals(IEnvironment other)
        {
            if (!(other is _2048Environment input)) return false;

            // Base checks
            if (Reward != input.Reward || Timestep != input.Timestep) return false;

            // Content check
            ref int r0 = ref Data[0];
            ref int r1 = ref input.Data[1];
            for (int i = 0; i < 16; i++)
                if (Unsafe.Add(ref r0, i) != Unsafe.Add(ref r1, i))
                    return false;
            return true;
        }

        /// <inheritdoc/>
        ~_2048Environment() => Allocator.Return(Data);

        /// <inheritdoc/>
        public void Dispose()
        {
            Allocator.Return(Data);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Returns the index of the free tiles in the current state
        /// </summary>
        [Pure]
        private Span<int> GetFreePositions()
        {
            Span<int> positions = Data.AsSpan(16, 16);
            ref int r0 = ref Data[0];
            ref int rp = ref positions.GetPinnableReference();
            int free = 0;

            for (int i = 0; i < 16; i++)
            {
                if (Unsafe.Add(ref r0, i) > 0) continue;
                Unsafe.Add(ref rp, free++) = i;
            }

            return positions.Slice(0, free);
        }
    }
}
