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
// Copyright © Jorge Nocedal, 1990
// http://users.eecs.northwestern.edu/~nocedal/
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

namespace NeuralNetworkNET.SupervisedLearning.Optimization.Dependencies
{
    /// <summary>
    ///   Status codes for the <see cref="BoundedBroydenFletcherGoldfarbShanno"/>
    ///   function optimizer
    /// </summary>
    public enum BoundedBroydenFletcherGoldfarbShannoStatus
    {
        /// <summary>
        ///   The optimization stopped before convergence; maximum
        ///   number of iterations could have been reached
        /// </summary>
        Stop,

        /// <summary>
        ///   Maximum number of iterations was reached
        /// </summary>
        MaximumIterations,

        /// <summary>
        ///   The function output converged to a static 
        ///   value within the desired precision
        /// </summary>
        FunctionConvergence,

        /// <summary>
        ///   The function gradient converged to a minimum
        ///   value within the desired precision
        /// </summary>
        GradientConvergence,

        /// <summary>
        ///   The inner line search function failed. This could be an indication 
        ///   that there might be something wrong with the gradient function
        /// </summary>
        LineSearchFailed = -1,
    }
}