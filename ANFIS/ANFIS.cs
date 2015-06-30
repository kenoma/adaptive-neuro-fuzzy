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
        
        private ITerm[] terms;

        /// <summary>
        /// Consequent parts of rules
        /// </summary>
        private double[][] z;

        public ANFIS(ITerm[] InitialTermSet)
        {
            if (InitialTermSet == null || InitialTermSet.Length == 0)
                throw new Exception("Ruleset is empty");
            this.numOfRules = InitialTermSet.Length;
            terms = InitialTermSet;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x">Input values</param>
        /// <param name="y">Desired output</param>
        public void TrainAFIS(double[][] x, double[][] y, ITraining TrainingAlgo)
        {
            int epoch = 0;
            do
            {
                double Error = TrainingAlgo.Iteration(x, y, z, terms);
                Console.WriteLine("[{0}] Epoch {1}, Error {2}\t", DateTime.Now, epoch++, Error);

            } while (!TrainingAlgo.isTrainingstoped());
            Console.WriteLine("Training done");
        }


        public double[] Inference(double[] x)
        {
            if (x.Length != inputDim)
                throw new Exception("Wrong input dimension");

            return Inference(x, z, terms, numOfRules, outputDim);
        }

        public static double[] Inference(double[] x, double[][] z, ITerm[] terms, int NumOfRules, int OutputDim)
        {
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
                    y[C] += firings[i] / firingSum * z[i][C];


            return y;
        }

    }
}
