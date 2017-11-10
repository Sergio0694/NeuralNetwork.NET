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

using NeuralNetworkNET.SupervisedLearning.Optimization.Abstract;
using NeuralNetworkNET.SupervisedLearning.Optimization.Dependencies;

namespace NeuralNetworkNET.SupervisedLearning.Optimization
{
    /// <summary>
    ///   Gradient Descent (GD) for unconstrained optimization
    /// </summary>
    public class GradientDescent : GradientOptimizationMethodBase
    {
        // Gets the current convergence instance to monitor the progress
        private readonly RelativeConvergence Convergence = null;
        
        // Optimization parameter during the minimization
        private readonly int NumberOfUpdatesBeforeConvergenceCheck = 1;

        /// <summary>
        ///   Gets or sets the maximum change in the average log-likelihood
        ///   after an iteration of the algorithm used to detect convergence.
        ///   Default is 1e-5
        /// </summary> 
        public double Tolerance
        {
            get => Convergence.Tolerance;
            set => Convergence.Tolerance = value;
        }

        /// <summary>
        ///   Creates a new instance of the GD optimization algorithm
        /// </summary>
        public GradientDescent()
        {
            Tolerance = 1e-5;
        }

        // The current learning rate index
        private int _ToleranceIndex;

        // Private vector with the range of learning rates to use
        private readonly double[] ToleranceVector = { 1e-3, 2e-3, 5e-3, 1e-2 };

        /// <summary>
        ///   Implements the actual optimization algorithm. This
        ///   method should try to minimize the objective function
        /// </summary>
        protected override bool Optimize()
        {
            int updates = 0;
            while (true)
            {
                // Check the cancellation
                if (Token.IsCancellationRequested) break;

                // Perform the gradient descent
                double[] gradient = Gradient(Solution);
                double rate = ToleranceVector[_ToleranceIndex];
                for (int i = 0; i < Solution.Length; i++)
                    Solution[i] -= rate * gradient[i];

                // Check the progress
                updates++;
                if (updates >= NumberOfUpdatesBeforeConvergenceCheck)
                {
                   // Convergence.NewValue = Function(Solution);
                    updates = 0;
                }

                // Convergence check
                if (Convergence.HasConverged)
                {
                    if (_ToleranceIndex < ToleranceVector.Length - 1) _ToleranceIndex++;
                    else break;
                }
                else if (_ToleranceIndex > 0) _ToleranceIndex--;
            }
            return true;
        }
    }
}
