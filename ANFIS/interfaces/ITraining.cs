using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANFIS
{
    public interface ITraining
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="x">Input</param>
        /// <param name="y">Desired output</param>
        /// <param name="z">consequent parts of rules</param>
        /// <param name="terms">Rulebase</param>
        /// <returns>Iteration Error</returns>
        double Iteration(double[][] x, double[][] y, double[][] z, ITerm[] terms);
        double Error(double[][] x, double[][] y, double[][] z, ITerm[] terms);
        bool isTrainingstoped();
    }
}
