using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroFuzzy
{
    public interface IRuleExtractor
    {
        void ExtractRules(double[][] input, double[][] output, out double[][] RuleCentroids, out double[][] RuleConsequences);
    }
}
