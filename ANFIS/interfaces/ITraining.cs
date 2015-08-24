using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroFuzzy
{
    public delegate void UnknownCase(IList<IRule> RuleBase, double[] centroid, double[] consequence, double[] neighbourhood);
    public interface ITraining
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="x">Input</param>
        /// <param name="y">Desired output</param>
        /// <param name="z">consequent parts of rules</param>
        /// <param name="RuleBase">Set of rules with membership function parameters and consequences</param>
        /// <returns>Iteration Error</returns>
        double Iteration(double[][] x, double[][] y, IList<IRule> RuleBase);

        /// <summary>
        /// Overall error of FIS
        /// </summary>
        /// <param name="x">inputs</param>
        /// <param name="y">desired outputs</param>
        /// <param name="RuleBase">Rule set of FIS</param>
        /// <returns></returns>
        double Error(double[][] x, double[][] y, IList<IRule> RuleBase);

        /// <summary>
        /// Event to manage situation when no one rule from database is sure for specific input
        /// </summary>
        event UnknownCase UnknownCaseFaced;

        /// <summary>
        /// Inform about training process
        /// </summary>
        /// <returns></returns>
        bool isTrainingstoped();

        /// <summary>
        /// Informs if during training we are managing unknown cases
        /// </summary>
        /// <returns></returns>
        bool isAdjustingRules();
    }
}
