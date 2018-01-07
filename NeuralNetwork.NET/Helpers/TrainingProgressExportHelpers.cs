using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Results;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A class with some helper methods to quickly convert a training report to a 2D chart
    /// </summary>
    public static class TrainingProgressExportHelpers
    {
        // The Python matplotlib template
        private static readonly String PyTemplate = @"import matplotlib.pyplot as plt
                                                      x = [$VALUES$]
                                                      plt.grid(linestyle=""dashed"")
                                                      plt.ylabel(""$YLABEL$"")
                                                      plt.xlabel(""Epoch"")
                                                      plt.plot(x)
                                                      plt.show()".Split(new[] { Environment.NewLine }, StringSplitOptions.None).Aggregate(String.Empty, (s, l) => $"{s}{l.Trim()}{Environment.NewLine}");

        // The custom 4-spaces indentation for the data points (the \t character is not consistent across different editors)
        private const String Tab = "    ";

        /// <summary>
        /// Returns a Python script to plot a 2D chart from the given progress reports
        /// </summary>
        /// <param name="results">The input results to plot</param>
        /// <param name="type">The type of progress report to plot</param>
        public static String AsPythonMatplotlibChart([NotNull] this IReadOnlyList<DatasetEvaluationResult> results, TrainingReportType type)
        {
            // Result value extractor
            float GetResultValue(DatasetEvaluationResult result)
            {
                switch (type)
                {
                    case TrainingReportType.Accuracy: return result.Accuracy;
                    case TrainingReportType.Cost: return result.Cost;
                    default: throw new ArgumentOutOfRangeException(nameof(type), "Invalid report type");
                }
            }

            // Load the template and extract the values to plot
            String
                values = results
                    .Select((r, i) => (Result: r, Index: i))
                    .Aggregate($"{Environment.NewLine}{Tab}", (b, v) =>
                    {
                        String separator = v.Index == results.Count - 1 ? Environment.NewLine : $",{Environment.NewLine}{Tab}";
                        return $"{b}{GetResultValue(v.Result).ToString(CultureInfo.InvariantCulture)}{separator}";
                    }),
                ylabel = type == TrainingReportType.Accuracy ? "Accuracy" : "Cost";
            return PyTemplate.Replace("$VALUES$", values).Replace("$YLABEL$", ylabel);
        }
    }
}
