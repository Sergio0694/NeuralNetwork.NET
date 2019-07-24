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
            GetFreePositionReference(Data) = ThreadSafeRandom.NextInt(1, 3) * 2;
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
            Data.AsSpan(0, 16).CopyTo(data);
            ref var rdata = ref data[0];
            ref var rx = ref action == RIGHT ? ref Descending[0] : ref Ascending[0];
            ref var ry = ref action == DOWN ? ref Descending[0] : ref Ascending[0];
            var direction = Directions[action];
            var score = 0;
            var moved = false;
            var won = false;

            // Iterate on all the cells in the grid
            for (var i = 0; i < 4; i++)
            {
                for (var j = 0; j < 4; j++)
                {
                    // Get the current cell, skip if it's empty
                    var x = Unsafe.Add(ref rx, i);
                    var y = Unsafe.Add(ref ry, j);
                    ref var rxy = ref Unsafe.Add(ref rdata, y * 4 + x);
                    if (rxy == 0) continue;

                    // Find the farthest cell with a value, and the farthest empty cell
                    var (tx, ty) = (x, y);
                    ref var rtxy0 = ref rxy;
                    ref var rtxy1 = ref rxy;
                    do
                    {
                        rtxy1 = ref rtxy0;
                        tx += direction.X;
                        ty += direction.Y;
                    } while (tx >= 0 && tx < 4 &&
                             ty >= 0 && ty < 4 &&
                             (rtxy0 = ref Unsafe.Add(ref rdata, ty * 4 + tx)) == 0);

                    // Check if the current cell can be moved
                    if (Unsafe.AreSame(ref rxy, ref rtxy0)) continue;
                    ref var rmaptxy = ref Unsafe.Add(ref rmap, ty * 4 + tx);
                    if (rtxy0 == rxy && !rmaptxy)
                    {
                        // Merge if the farthest cell has the same value and it hadn't been merged before
                        rmaptxy = true;
                        rtxy0 *= 2;
                        rxy = 0;
                        score += rtxy0;
                        moved = true;
                        if (rtxy0 == 2048) won = true;
                    }
                    else if (rtxy1 == 0 && !Unsafe.AreSame(ref rxy, ref rtxy1))
                    {
                        // If the farthest cell is empty, move the current cell
                        rtxy1 = rxy;
                        rxy = 0;
                        moved = true;
                    }
                }
            }

            // If at least one cell has been moved, insert a new random cell in an empty space
            if (moved && !won)
            {
                ref var rfree = ref GetFreePositionReference(data);
                if (rfree == 0) rfree = ThreadSafeRandom.NextInt(1, 3) * 2;
            }

            score += Reward;
            var timestamp = Timestep + (moved ? 1 : 0);
            return new _2048Environment(data, score, timestamp);
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
            var data = Allocator.Rent(16);
            Data.AsSpan(0, Size).CopyTo(data.AsSpan(0, Size));
            var instance = new _2048Environment(data, Reward, Timestep);
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
        /// <param name="data">The input state to analyze</param>
        [Pure]
        private static ref int GetFreePositionReference([NotNull] int[] data)
        {
            Span<int> positions = stackalloc int[16];
            ref int rp = ref positions.GetPinnableReference();
            ref int r0 = ref data[0];
            int free = 0;

            // Iterate over the current state and mark the free cells
            for (int i = 0; i < 16; i++)
            {
                if (Unsafe.Add(ref r0, i) > 0) continue;
                Unsafe.Add(ref rp, free++) = i;
            }

            // Pick a random free cell and return a reference to it
            if (free == 0) return ref r0;
            int
                pick = ThreadSafeRandom.NextInt(max: free),
                index = Unsafe.Add(ref rp, pick);
            return ref Unsafe.Add(ref r0, index);
        }
    }
}
