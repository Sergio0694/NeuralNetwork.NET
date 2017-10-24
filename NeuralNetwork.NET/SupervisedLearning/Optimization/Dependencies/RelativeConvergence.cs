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

namespace NeuralNetworkNET.SupervisedLearning.Optimization.Dependencies
{
    /// <summary>
    ///   Relative convergence criteria.
    /// </summary>
    /// 
    /// <remarks>
    ///   This class can be used to track progress and convergence
    ///   of methods which rely on the relative change of a value.
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
    ///   double value = criteria.OldValue; // 1234.56
    /// </code>
    /// </example>
    public class RelativeConvergence
    {
        private double _Tolerance;

        /// <summary>
        ///   Gets or sets the maximum relative change in the watched value
        ///   after an iteration of the algorithm used to detect convergence.
        ///   Default is zero
        /// </summary>
        public double Tolerance
        {
            get => _Tolerance;
            set => _Tolerance = value < 0 ? throw new ArgumentOutOfRangeException("value", "Tolerance should be positive") : value;
        }

        private int _MaxIterations = 100;

        /// <summary>
        ///   Gets or sets the maximum number of iterations
        ///   performed by the iterative algorithm. 
        ///   Default is 100
        /// </summary>
        public int MaxIterations
        {
            get => _MaxIterations;
            set => _MaxIterations = value < 0 ? throw new ArgumentOutOfRangeException("value", "The maximum number of iterations should be positive") : value;
        }

        // The initial value for the function to minimize
        private readonly double StartValue;

        // The maximum number of consecutive function evaluations
        private readonly int MaxChecks;

        /// <summary>
        ///   Initializes a new instance of the <see cref="RelativeConvergence"/> class
        /// </summary>
        public RelativeConvergence()
        {
            MaxIterations = 100;
            Tolerance = 0;
            MaxChecks = 1;
            StartValue = 0;
            Clear();
        }

        /// <summary>
        ///   Gets or sets the watched value before the iteration
        /// </summary>
        public double OldValue { get; private set; }

        private double _NewValue;

        /// <summary>
        ///   Gets or sets the watched value after the iteration
        /// </summary>
        public double NewValue
        {
            get => _NewValue;
            set
            {
                OldValue = _NewValue;
                _NewValue = value;
                CurrentIteration++;
            }
        }

        /// <summary>
        ///   Gets the current iteration number
        /// </summary>
        public int CurrentIteration { get; set; }

        // Local counter
        private int _Checks;

        /// <summary>
        ///   Gets whether the algorithm has converged
        /// </summary>
        public bool HasConverged
        {
            get
            {
                bool converged = CheckConvergence();
                _Checks = converged ? _Checks + 1 : 0;
                return _Checks >= MaxChecks;
            }
        }

        // Simple function that checks the function convergence given the current parameters
        private bool CheckConvergence()
        {
            // Iterations count
            if (MaxIterations > 0 && CurrentIteration >= MaxIterations) return true;

            // Stopping criteria is likelihood convergence
            if (Tolerance > 0)
            {
                if (Delta <= Tolerance * Math.Abs(OldValue)) return true;
            }

            // Check if we have reached an invalid or perfectly separable answer
            return double.IsNaN(NewValue) || double.IsInfinity(NewValue);
        }

        /// <summary>
        ///   Gets the absolute difference between the <see cref="NewValue"/> and <see cref="OldValue"/>
        ///   as as <c>Math.Abs(OldValue - NewValue)</c>
        /// </summary>
        public double Delta => Math.Abs(OldValue - NewValue);

        /// <summary>
        ///   Resets this instance, reverting all iteration statistics
        ///   statistics (number of iterations, last error) back to zero
        /// </summary>
        public void Clear()
        {
            NewValue = StartValue;
            OldValue = StartValue;
            CurrentIteration = 0;
            _Checks = 0;
        }
    }
}
