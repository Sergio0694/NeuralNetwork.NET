using System;
using System.Diagnostics;

namespace NeuralNetworkNET.NUnit
{
    /// <summary>
    /// A static class with some helper methods for the library tests
    /// </summary>
    internal static class NUnitHelpers
    {
        /// <summary>
        /// Checks that the input test throws the required exception
        /// </summary>
        /// <typeparam name="T">The target exception type</typeparam>
        /// <param name="test">The test to run</param>
        public static void AssertThrows<T>(Action test) where T : Exception
        {
            bool passed;
            try
            {
                test();
                passed = false;
            }
            catch (T)
            {
                passed = true;
            }
            catch
            {
                passed = false;
            }
            Debug.Assert(passed);
        }
    }
}
