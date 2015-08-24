using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroFuzzy
{
    /// <summary>
    /// Interface for rule extraction
    /// </summary>
    public interface IRuleExtractor
    {
        /// <summary>
        /// Core method for rule extraction. As input it takes an input array with desired outputs. 
        /// </summary>
        /// <param name="input">Inputs</param>
        /// <param name="output">Desired ouputs</param>
        /// <param name="RuleCentroids">'Row' rule centroids</param>
        /// <param name="RuleConsequences">'Row' rule consequences</param>
        void ExtractRules(double[][] input, double[][] output, out double[][] RuleCentroids, out double[][] RuleConsequences);
    }
}
