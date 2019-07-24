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
        private static readonly ArrayPool<int> Allocator = ArrayPool<int>.Create(16, 1000);

        /// <summary>
        /// The array representing the environment current state
        /// </summary>
        [NotNull]
        public readonly int[] Data;

        /// <summary>
        /// Creates a new, empty <see cref="TicTacToeEnvironment"/> instance
        /// </summary>
        public _2048Environment()
        {
            Data = Allocator.Rent(16);
            Data[ThreadSafeRandom.NextInt(max: 4) * 4 + ThreadSafeRandom.NextInt(max: 4)] = ThreadSafeRandom.NextInt(1, 3) * 2;
            GetFreePositionReference() = ThreadSafeRandom.NextInt(1, 3) * 2;
        }

        private _2048Environment([NotNull] int[] data, int reward, int timestep)
        {
            Data = data;
            Reward = reward;
            Timestep = timestep;
        }

        private static readonly (int X, int Y)[] Directions =
        {
            (0, -1),    // Up
            (0, 1),     // Down
            (-1, 0),    // Left
            (1, 0)      // Right
        };

        private static readonly int[] Ascending = { 0, 1, 2, 3 };
        private static readonly int[] Descending = { 3, 2, 1, 0 };

        private const int UP = 0;
        private const int DOWN = 1;
        private const int LEFT = 2;
        private const int RIGHT = 3;

        /// <inheritdoc/>
        public IEnvironment Execute(int action)
        {
            Span<bool> map = stackalloc bool[16];
            ref var rmap = ref map.GetPinnableReference();
            var data = Allocator.Rent(16);
            ref var rdata = ref data[0];
            ref var rx = ref action == RIGHT ? ref Descending[0] : ref Ascending[0];
            ref var ry = ref action == DOWN ? ref Descending[1] : ref Ascending[1];
            var direction = Directions[action];
            var score = 0;

            for (var i = 1; i < 4; i++)
            {
                for (var j = 1; j < 4; j++)
                {
                    var x = Unsafe.Add(ref rx, i);
                    var y = Unsafe.Add(ref ry, j);
                    ref var rxy = ref Unsafe.Add(ref rdata, y * 4 + x);
                    if (rxy == 0) continue;

                    var (tx, ty) = (x, y);
                    ref var rtxy = ref rxy;
                    do
                    {
                        tx += direction.X;
                        ty += direction.Y;
                    } while (tx >= 0 && tx < 4 &&
                             ty >= 0 && ty < 4 &&
                             (rtxy = ref Unsafe.Add(ref rdata, ty * 4 + tx)) == 0);

                    if (Unsafe.AreSame(ref rxy, ref rtxy)) continue;
                    if (rtxy == 0)
                    {
                        rtxy = rxy;
                        rxy = 0;
                    }
                    else if (rtxy == rxy)
                    {
                        ref var rmaptxy = ref Unsafe.Add(ref rmap, ty * 4 + tx);
                        if (rmaptxy) continue;
                        rmaptxy = true;
                        rtxy *= 2;
                        rxy = 0;
                        score += rtxy;
                    }
                }
            }

            return new _2048Environment(data, Reward + score, Timestep + 1);
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
        private ref int GetFreePositionReference()
        {
            Span<int> positions = stackalloc int[16];
            ref int r0 = ref Data[0];
            ref int rp = ref positions.GetPinnableReference();
            int free = 0;

            // Iterate over the current state and mark the free cells
            for (int i = 0; i < 16; i++)
            {
                if (Unsafe.Add(ref r0, i) > 0) continue;
                Unsafe.Add(ref rp, free++) = i;
            }

            // Pick a random free cell and return a reference to it
            int
                pick = ThreadSafeRandom.NextInt(max: free),
                index = Unsafe.Add(ref rp, pick);
            return ref Unsafe.Add(ref r0, index);
        }
    }
}
