// ===================================
// Credits for the base of this code
// ===================================
//
// Accord Math Library
// The Accord.NET Framework
// http://accord-framework.net
//
// Copyright © César Souza, 2009-2017
// cesarsouza at gmail.com
//
//    This library is free software; you can redistribute it and/or
//    modify it under the terms of the GNU Lesser General Public
//    License as published by the Free Software Foundation; either
//    version 2.1 of the License, or (at your option) any later version.
//
//    This library is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//    Lesser General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public
//    License along with this library; if not, write to the Free Software
//    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

using System;
using System.Collections.Generic;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.SupervisedLearning.Trackers
{
    /// <summary>
    ///   Relative convergence criteria
    /// </summary>
    /// 
    /// <remarks>
    ///   This class can be used to track progress and convergence
    ///   of methods which rely on the relative change of a value
    /// </remarks>
    /// 
    /// <example>
    /// <code>
    ///   // Create a new convergence criteria with unlimited iterations
    ///   var criteria = new RelativeConvergence(iterations: 0, tolerance: 0.1);
    ///   
    ///   int progress = 1;
    ///   
    ///   do
    ///   {
    ///       // Do some processing...
    ///   
    ///   
    ///       // Update current iteration information:
    ///       criteria.NewValue = 12345.6 / progress++;
    ///   
    ///   } while (!criteria.HasConverged);
    ///   
    ///   
    ///   // The method will converge after reaching the 
    ///   // maximum of 11 iterations with a final value
    ///   // of 1234.56:
    ///   
    ///   int iterations = criteria.CurrentIteration; // 11
    ///   float value = criteria.OldValue; // 1234.56
    /// </code>
    /// </example>
    internal class RelativeConvergence
    {
        private float _Tolerance;

        /// <summary>
        /// Gets or sets the maximum relative change in the watched value
        /// after an iteration of the algorithm used to detect convergence
        /// </summary>
        public float Tolerance
        {
            get => _Tolerance;
            set => _Tolerance = value < 0 ? throw new ArgumentOutOfRangeException(nameof(value), "Tolerance should be positive") : value;
        }

        /// <summary>
        /// Gets the size of the convergence window
        /// </summary>
        private readonly int ConvergenceWindow;

        /// <summary>
        /// Initializes a new instance of the <see cref="RelativeConvergence"/> class
        /// </summary>
        public RelativeConvergence(float tolerance, int window)
        {
            if (tolerance <= 0) throw new ArgumentOutOfRangeException(nameof(tolerance), "The tolerance must be a positive value");
            if (window < 1) throw new ArgumentOutOfRangeException(nameof(window), "The tolerance window must be at least equal to 1");
            Tolerance = tolerance;
            ConvergenceWindow = window;
        }

        // The previous value for the convergence check
        private readonly Queue<float> _PreviousValues = new Queue<float>();

        private float _Value;

        /// <summary>
        /// Gets or sets the watched value after the iteration
        /// </summary>
        public float Value
        {
            get => _Value;
            set
            {
                if (_PreviousValues.Count == ConvergenceWindow) _PreviousValues.Dequeue();
                _PreviousValues.Enqueue(value);
                _Value = value;
            }
        }

        /// <summary>
        /// Gets whether the algorithm has converged
        /// </summary>
        public bool HasConverged
        {
            get
            {
                if (_PreviousValues.Count < ConvergenceWindow) return false;
                float[] values = _PreviousValues.ToArray();
                float min = float.MaxValue, max = float.MinValue;
                unsafe
                {
                    fixed (float* p = values)
                    {
                        for (int i = 0; i < values.Length; i++)
                        {
                            float value = p[i];
                            if (value > max) max = value;
                            if (value < min) min = value;
                        }
                    }
                }
                return (max - min).Abs() < Tolerance;
            }
        }
    }
}
