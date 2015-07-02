using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANFIS.rextractors
{
    public class KMEANSExtractor : IRuleExtractor
    {
        /// <summary>
        /// before clustering each output vector appended to input vector
        /// after clustering we obtain initial rule guesses
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <param name="RuleNumbers"></param>
        /// <param name="RuleCentroids"></param>
        /// <param name="RuleConsequences"></param>
        public void ExtractRules(double[][] input, double[][] output, int RuleNumbers, out double[][] RuleCentroids, out double[][] RuleConsequences)
        {
            double[][] x = new double[input.Length][];
            int inputLength = input[0].Length;
            int outputLength = output[0].Length;

            for (int row = 0; row < x.Length; row++)
            {
                x[row] = new double[inputLength+outputLength];
                Array.Copy(input[row], x[row], inputLength);
                Array.Copy(output[row], 0, x[row], input[row].Length, output[row].Length);
            }
            double[][] c = kmeans.clustering(x, RuleNumbers, 3, kmeansType.Forgy);
            RuleCentroids = new double[RuleNumbers][];
            RuleConsequences = new double[RuleNumbers][];
            for (int row = 0; row < RuleNumbers; row++)
            {
                RuleCentroids[row] = new double[inputLength];
                RuleConsequences[row] = new double[outputLength];
                Array.Copy(c[row], RuleCentroids[row], inputLength);
                Array.Copy(c[row], inputLength, RuleConsequences[row], 0, outputLength);
            }
        }
    }
}
