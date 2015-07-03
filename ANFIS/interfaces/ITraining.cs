using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANFIS
{
    public delegate void AdjustRuleBase(IList<IRule> RuleBase, double[] centroid, double[] consequence, double[] neighbourhood);
    public interface ITraining
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="x">Input</param>
        /// <param name="y">Desired output</param>
        /// <param name="z">consequent parts of rules</param>
        /// <param name="RuleBase">Set of rules - i.e. membership function parameters+consequence part of rule</param>
        /// <returns>Iteration Error</returns>
        double Iteration(double[][] x, double[][] y, IList<IRule> RuleBase);
        double Error(double[][] x, double[][] y, IList<IRule> RuleBase);
        event AdjustRuleBase AddRule;
        bool isTrainingstoped();
        bool isAdjustingRules();
    }
}
