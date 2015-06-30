﻿using System;
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
        double threshold = 0.0;
        bool isStop = false;

        public BackpropTraining(double LearningRate, double threshold)
        {
            this.learningRate = LearningRate;
            this.threshold = threshold;
        }

        public double Iteration(double[][] x, double[][] y, double[][] z, ITerm[] terms)
        {
            if (z == null || z.Length == 0)
                throw new Exception("No consequence part of rules");
            if (x.Length != y.Length)
                throw new Exception("Input and desired output lengths not match");
            if (terms == null || terms.Length != z.Length)
                throw new Exception("Incorrect rulebase");

            int outputDim = z[0].Length;
            int numOfRules = terms.Length;

            double globalError = 0.0;



            double[] firings = new double[numOfRules];

            for (int sample = 0; sample < x.Length; sample++)
            {
                double[] o = new double[outputDim];
                double firingSum = 0.0;

                for (int i = 0; i < numOfRules; i++)
                {
                    firings[i] = terms[i].Membership(x[sample]);
                    firingSum += firings[i];
                }

                for (int i = 0; i < numOfRules; i++)
                    for (int C = 0; C < outputDim; C++)
                        o[C] += firings[i] / firingSum * z[i][C];

                for (int rule = 0; rule < terms.Length; rule++)
                {
                    double[] parm = terms[rule].Parameters;
                    double[] grad = terms[rule].GetGradient(x[sample]);

                    for (int p = 0; p < parm.Length; p++)
                    {
                        double g = 0.0;

                        for (int C = 0; C < outputDim; C++)
                        {
                            double subSum = 0.0;
                            for (int i = 0; i < numOfRules; i++)
                                subSum += (i == rule ?
                                    (grad[p] * (1.0 / firingSum - firings[rule] / (firingSum * firingSum))) :
                                    (-firings[rule] * grad[p] / (firingSum * firingSum))) * z[i][C];


                            g += (o[C] - y[sample][C]) * subSum;
                        }
                        
                        parm[p] -= learningRate * g;
                    }
                }

                for (int i = 0; i < numOfRules; i++)
                    for (int C = 0; C < outputDim; C++)
                        z[i][C] -= learningRate * (o[C] - y[sample][C]) * firings[i] / firingSum;

                for (int C = 0; C < outputDim; C++)
                    globalError += Math.Abs(o[C] - y[sample][C]);

            }

            if (Math.Abs(lastError - globalError) < threshold)
                isStop = true;
            else
                isStop = false;
            lastError = globalError;

            return globalError / x.Length;
        }

        public bool isTrainingstoped()
        {
            return isStop;
        }


        public double Error(double[][] x, double[][] y, double[][] z, ITerm[] terms)
        {
            if (z == null || z.Length == 0)
                throw new Exception("No consequence part of rules");
            if (x.Length != y.Length)
                throw new Exception("Input and desired output lengths not match");
            if (terms == null || terms.Length != z.Length)
                throw new Exception("Incorrect rulebase");

            int outputDim = z[0].Length;
            int numOfRules = terms.Length;

            double globalError = 0.0;

            for (int sample = 0; sample < x.Length; sample++)
            {
                double[] o = ANFIS.Inference(x[sample], z, terms, numOfRules, outputDim);
                for (int C = 0; C < outputDim; C++)
                    globalError += Math.Abs(o[C] - y[sample][C]);
            }

            return globalError / x.Length;
        }
    }
}