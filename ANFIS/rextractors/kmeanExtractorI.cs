using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuroFuzzy.misc;
namespace NeuroFuzzy.rextractors
{
    public class KMEANSExtractorI : IRuleExtractor
    {

        public int RuleNumbers { get; set; }

        public KMEANSExtractorI(int RuleNumbers)
        {
            this.RuleNumbers = RuleNumbers;
        }

        /// <summary>
        /// clustering made only for input, after this consequences are made as averaged of clustering
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <param name="RuleNumbers"></param>
        /// <param name="RuleCentroids"></param>
        /// <param name="RuleConsequences"></param>
        public void ExtractRules(double[][] input, double[][] output, out double[][] RuleCentroids, out double[][] RuleConsequences)
        {
            double[][] x = new double[input.Length][];
            int inputLength = input[0].Length;
            int outputLength = output[0].Length;

            for (int row = 0; row < x.Length; row++)
            {
                x[row] = new double[inputLength];
                Array.Copy(input[row], x[row], inputLength);
            }
            double[][] c = kmeans.clustering(x, RuleNumbers, 3, kmeansType.kmeanspp);

            int[] assign = new int[x.Length];
            for (int row = 0; row < x.Length; row++)
                assign[row] = math.DetectBucket(x[row], c);

            RuleCentroids = new double[RuleNumbers][];
            RuleConsequences = new double[RuleNumbers][];

            for (int row = 0; row < RuleNumbers; row++)
            {
                RuleCentroids[row] = new double[inputLength];

                Array.Copy(c[row], RuleCentroids[row], inputLength);

                RuleConsequences[row] = new double[outputLength];

                var cluster = output.Select((z, i) => new { Z = z, I = assign[i] }).Where(z => z.I == row).ToArray();
                for (int i = 0; i < outputLength; i++)
                    RuleConsequences[row][i] = cluster.Average(z => z.Z[i]);

            }
        }
    }
}
