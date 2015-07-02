using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANFIS.training
{
    public class BackpropTraining : ITraining
    {
        double learningRate = 1e-10;
        double lastError = double.MaxValue;
        double abstol,reltol;
        bool isStop = false;

        public BackpropTraining(double LearningRate, double abstol = 1e-4, double reltol = 1e-5)
        {
            this.learningRate = LearningRate;
            this.abstol = abstol;
            this.reltol = reltol;
        }

        public double Iteration(double[][] x, double[][] y,  IRule[] ruleBase)
        {
            isStop = false;
            
            if (x.Length != y.Length)
                throw new Exception("Input and desired output lengths not match");
            if (ruleBase == null || ruleBase.Length == 0)
                throw new Exception("Incorrect rulebase");

            int outputDim = ruleBase[0].Z.Length;
            int numOfRules = ruleBase.Length;

            double globalError = 0.0;



            double[] firings = new double[numOfRules];

            for (int sample = 0; sample < x.Length; sample++)
            {
                double[] o = new double[outputDim];
                double firingSum = 0.0;

                for (int i = 0; i < numOfRules; i++)
                {
                    firings[i] = ruleBase[i].Membership(x[sample]);
                    firingSum += firings[i];
                }
            
                for (int i = 0; i < numOfRules; i++)
                    for (int C = 0; C < outputDim; C++)
                        o[C] += firings[i] / firingSum * ruleBase[i].Z[C];

                for (int rule = 0; rule < ruleBase.Length; rule++)
                {
                    double[] parm = ruleBase[rule].Parameters;
                    double[] grad = ruleBase[rule].GetGradient(x[sample]);

                    for (int p = 0; p < parm.Length; p++)
                    {
                        double g = dEdP(y[sample], o, ruleBase, firings, grad, firingSum, rule, outputDim, numOfRules, p);

                        parm[p] -= learningRate * g;
                    }
                }

                for (int i = 0; i < numOfRules; i++)
                    for (int C = 0; C < outputDim; C++)
                        ruleBase[i].Z[C] -= learningRate * (o[C] - y[sample][C]) * firings[i] / firingSum;

                for (int C = 0; C < outputDim; C++)
                    globalError += Math.Abs(o[C] - y[sample][C]);
                if (double.IsNaN(globalError))
                    Console.WriteLine("");

            }

            checkStop(globalError);

            return globalError / x.Length;
        }

      

        private void checkStop(double globalError)
        {
            if (globalError < abstol)
                isStop = true;

            if (Math.Abs(lastError - globalError) < reltol)
                isStop = true;

            lastError = globalError;
        }

        private static double dEdP(double[] y, double[] o,
            IRule[] z, 
            double[] firings, 
            double[] grad, 
            double firingSum, 
            int rule, 
            int outputDim, 
            int numOfRules, 
            int p)
        {
            double g = 0.0;

            for (int C = 0; C < outputDim; C++)
            {
                double subSum = 0.0;
                for (int i = 0; i < numOfRules; i++)
                    subSum += (i == rule ?
                        (grad[p] * (1.0 / firingSum - firings[rule] / (firingSum * firingSum))) :
                        (-firings[i] * grad[p] / (firingSum * firingSum))) * z[i].Z[C];


                g += (o[C] - y[C]) * subSum;
            }
            return g;
        }

        public bool isTrainingstoped()
        {
            return isStop;
        }


        public double Error(double[][] x, double[][] y, IRule[] ruleBase)
        {
            
            if (x.Length != y.Length)
                throw new Exception("Input and desired output lengths not match");
            if (ruleBase == null || ruleBase.Length == 0)
                throw new Exception("Incorrect rulebase");

            int outputDim = ruleBase[0].Z.Length;
            int numOfRules = ruleBase.Length;

            double globalError = 0.0;

            for (int sample = 0; sample < x.Length; sample++)
            {
                double[] o = ANFIS.Inference(x[sample], ruleBase);
                for (int C = 0; C < outputDim; C++)
                    globalError += Math.Abs(o[C] - y[sample][C]);
            }

            return globalError / x.Length;
        }
    }
}
