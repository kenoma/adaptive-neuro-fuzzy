using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANFIS
{
    public interface IRuleExtractor
    {
        void ExtractRules(double[][] input, double[][] output, int RuleNumbers, out double[][] RuleCentroids, out double[][] RuleConsequences);
    }
}
