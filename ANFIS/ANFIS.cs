using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroFuzzy
{

    public class ANFIS
    {
        private int inputDim, outputDim, numOfRules;
        
        private IRule[] ruleBase;

        public IRule[] RuleBase { get { return ruleBase; } set { ruleBase = value; } }

        public ANFIS(IList<IRule> RuleSet)
        {
            if (RuleSet == null || RuleSet.Count == 0)
                throw new Exception("Ruleset is empty");
            this.numOfRules = RuleSet.Count;
            ruleBase = RuleSet.ToArray();
        }

        public double[] Inference(double[] x)
        {
            return Inference(x, ruleBase);
        }

        public static double[] Inference(double[] x, IList<IRule> RuleBase)
        {
            int NumOfRules = RuleBase.Count;

            int OutputDim = RuleBase[0].Z.Length;

            double[] firings = new double[NumOfRules];
            double[] y = new double[OutputDim];
            double firingSum = 0.0;

            for (int i = 0; i < NumOfRules; i++)
            {
                firings[i] = RuleBase[i].Membership(x);
                firingSum += firings[i];
            }

            for (int i = 0; i < NumOfRules; i++)
                for (int C = 0; C < OutputDim; C++)
                    y[C] += firings[i] / firingSum * RuleBase[i].Z[C];


            return y;
        }

    }
}
