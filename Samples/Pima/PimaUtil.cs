using System;
using System.IO;
using System.Linq;

namespace cux.neuro
{
    public static class PimaUtil
    {

        public static void Normalize(float[][] data)
        {
            float[] maxes = new float[data[0].Length];
            foreach (var tab in data)
            {
                for (int i = 0; i < tab.Length; i++)
                {
                    if (tab[i] > maxes[i])
                        maxes[i] = tab[i];
                }
            }
            foreach (var tab in data)
            {
                for (int i = 0; i < tab.Length; i++)
                {
                    tab[i] /= maxes[i];
                }
            }
        }

        public static (float[] First, float[] Second)[] LoadFromFile(string path)
        {
            string csvText = File.ReadAllText(path);
            string[] csvLines = csvText.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries);
            csvLines = csvLines.Skip(1).ToArray(); // remove header

            // parse CSV cells into 2d array of floats
            float[][] originalData = csvLines
                .Select(s => s.Split(',')
                .Select(float.Parse).ToArray())
                .ToArray();

            // normalize and filter the data
            Normalize(originalData);
            originalData = originalData.Where(FilterData).ToArray();

            // input - medical data
            float[][] input = originalData
                .Select(row => row[..^1])
                .ToArray();

            // output - verdict if a patient has diabetes
            float[][] output = originalData
                .Select(row => row.Last())
                .ToArray()
                .Select(HotEncode)
                .ToArray();

            return Enumerable.Zip(input, output).ToArray();
        }

        // remove empty fields
        private static bool FilterData(float[] d) => d[2] != 0 && d[3] != 0 && d[4] != 0;

        // hot encode single boolean value
        public static float[] HotEncode(float val) => new float[] { val, 1 - val };
    }
}
