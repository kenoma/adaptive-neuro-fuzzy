using NeuroFuzzy.misc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroFuzzy.rextractors
{

    /// <summary>
    /// Set rule base exactly as provided input
    /// </summary>
    public class OneToOneInitializercs : IRuleExtractor
    {
        public void ExtractRules(double[][] input, double[][] output, out double[][] RuleCentroids,
            out double[][] RuleConsequences)
        {
            var rx = new List<double[]>();
            var ry = new List<double[]>();
            for (var sample = 0; sample < input.Length; sample++)
            {
                var d = double.MaxValue;

                foreach (var xx in rx)
                    d = Math.Min(d, math.EuclidianDistance(xx, input[sample]));

                if (d > 1e-10)
                {
                    rx.Add(input[sample]);
                    ry.Add(output[sample]);
                }
            }

            RuleCentroids = rx.ToArray();
            RuleConsequences = ry.ToArray();
        }
    }
}
