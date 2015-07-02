using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANFIS
{
    /// <summary>
    /// 
    /// </summary>
    public class ANFIS
    {
        private int inputDim, outputDim, numOfRules;
        
        private IRule[] ruleBase;

        public IRule[] RuleBase { get { return ruleBase; } set { ruleBase = value; } }

        public ANFIS(IRule[] RuleSet)
        {
            if (RuleSet == null || RuleSet.Length == 0)
                throw new Exception("Ruleset is empty");
            this.numOfRules = RuleSet.Length;
            ruleBase = RuleSet;
        }

        public double[] Inference(double[] x)
        {
            if (x.Length != inputDim)
                throw new Exception("Wrong input dimension");

            return Inference(x, ruleBase);
        }

        public static double[] Inference(double[] x, IRule[] terms)
        {
            int NumOfRules = terms.Length;
            int OutputDim = terms[0].Z.Length;

            double[] firings = new double[NumOfRules];
            double[] y = new double[OutputDim];
            double firingSum = 0.0;

            for (int i = 0; i < NumOfRules; i++)
            {
                firings[i] = terms[i].Membership(x);
                firingSum += firings[i];
            }

            for (int i = 0; i < NumOfRules; i++)
                for (int C = 0; C < OutputDim; C++)
                    y[C] += firings[i] / firingSum * terms[i].Z[C];


            return y;
        }

    }
}
