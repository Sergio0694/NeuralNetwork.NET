using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
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
                    .Aggregate(String.Empty, (b, v) =>
                    {
                        String separator = v.Index == results.Count - 1 ? String.Empty : ",\n    ";
                        return $"{b}{GetResultValue(v.Result).ToString(CultureInfo.InvariantCulture)}{separator}";
                    }),
                ylabel = type == TrainingReportType.Accuracy ? "Accuracy" : "Cost",
                path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(Assembly.GetExecutingAssembly().Location)), "Assets", "PltTemplate.py"),
                template = File.ReadAllText(path);
            return template.Replace("$VALUES$", values).Replace("$YLABEL$", ylabel);
        }
    }
}
