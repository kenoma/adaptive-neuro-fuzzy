using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANFIS.training
{
    public class QPropTraining : ITraining
    {
        double etaPlus, etaMinus,deltaMax,deltaMin,defLRate;
        double lastError = double.MaxValue;
        double threshold = 0.0;

        bool isStop = false;

        public QPropTraining(double ErrorThreshold, double EtaPlus = 1.2, double EtaMinus = 0.5, double DeltaMax = 1, double DeltaMin = 1e-8, double InitialLearningRate = 1e-4)
        {
            this.etaMinus = EtaMinus;
            this.etaPlus = EtaPlus;
            this.deltaMax = DeltaMax;
            this.deltaMin = DeltaMin;
            this.threshold = ErrorThreshold;
            this.defLRate = InitialLearningRate;
        }


        private double[][] lRatesParams;
        private double[][] lRatesConseq;
        private double[][] prev_z_accum;
        private double[][] prev_p_accum;

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

            double[][] z_accum;
            double[][] p_accum;
            InitStuff(z, terms, outputDim, out z_accum, out p_accum);

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
                    //double[] parm = terms[rule].Parameters;
                    double[] grad = terms[rule].GetGradient(x[sample]);

                    for (int p = 0; p < grad.Length; p++)
                    {
                        double g = dEdP(y[sample], o, z, firings, grad, firingSum, rule, outputDim, numOfRules, p);
                        p_accum[rule][p] += g;
                    }
                }

                for (int i = 0; i < numOfRules; i++)
                    for (int C = 0; C < outputDim; C++)
                        z_accum[i][C] += (o[C] - y[sample][C]) * firings[i] / firingSum;

                for (int C = 0; C < outputDim; C++)
                    globalError += Math.Abs(o[C] - y[sample][C]);
            }

            updateLrates(p_accum, z_accum, prev_p_accum, prev_z_accum, lRatesParams, lRatesConseq);
            prev_p_accum = p_accum;
            prev_z_accum = z_accum;

            for (int rule = 0; rule < terms.Length; rule++)
            {
                double[] parm = terms[rule].Parameters;
                for (int p = 0; p < parm.Length; p++)
                    parm[p] -= lRatesParams[rule][p] * p_accum[rule][p];
            }

            for (int i = 0; i < numOfRules; i++)
                for (int C = 0; C < outputDim; C++)
                    z[i][C] -= lRatesConseq[i][C] * z_accum[i][C];


            checkStopCriteria(globalError);

            
            return globalError / x.Length;
        }

        private void checkStopCriteria(double globalError)
        {
            if (globalError < threshold)
                isStop = true;
            else
                isStop = false;
            lastError = globalError;
        }

        private void updateLrates(double[][] c_p_accum, double[][] c_z_accum, double[][] p_p_accum, double[][] p_z_accum, double[][] lr_p, double[][] lr_z)
        {
            if (p_p_accum == null || p_z_accum == null)
                return;

            for (int rule = 0; rule < c_p_accum.Length; rule++)
                for (int j = 0; j < c_p_accum[rule].Length; j++)
                {
                    double mltp = c_p_accum[rule][j] * p_p_accum[rule][j];
                    if (mltp > 0)
                    {
                        lr_p[rule][j] *= etaPlus;
                        lr_p[rule][j] = Math.Min(deltaMax, lr_p[rule][j]);
                    }
                    else if (mltp < 0)
                    {
                        lr_p[rule][j] *= etaMinus;
                        lr_p[rule][j] = Math.Max(deltaMin, lr_p[rule][j]);
                    }
                    else
                        lr_p[rule][j] = deltaMin;
                }


            for (int rule = 0; rule < c_z_accum.Length; rule++)
                for (int j = 0; j < c_z_accum[rule].Length; j++)
                {
                    double mltp = c_z_accum[rule][j] * p_z_accum[rule][j];
                    if (mltp > 0)
                    {
                        lr_z[rule][j] *= etaPlus;
                        lr_z[rule][j] = Math.Min(deltaMax, lr_z[rule][j]);
                    }
                    else if (mltp < 0)
                    {
                        lr_z[rule][j] *= etaMinus;
                        lr_z[rule][j] = Math.Max(deltaMin, lr_z[rule][j]);
                    }
                    else
                        lr_z[rule][j] = deltaMin;
                }
        }

        private void InitStuff(double[][] z, ITerm[] terms, int outputDim, out double[][] z_accum, out double[][] p_accum)
        {
            z_accum = new double[z.Length][];
            p_accum = new double[z.Length][];

            if (lRatesParams == null)
            {
                lRatesParams = new double[p_accum.Length][];
                for (int i = 0; i < z.Length; i++)
                {
                    lRatesParams[i] = new double[terms[i].Parameters.Length];
                    for (int j = 0; j < lRatesParams[i].Length; j++)
                        lRatesParams[i][j] = defLRate;
                }
            }
            if (lRatesConseq == null)
            {
                lRatesConseq = new double[z_accum.Length][];
                for (int i = 0; i < z.Length; i++)
                {
                    lRatesConseq[i] = new double[outputDim];
                    for (int j = 0; j < outputDim; j++)
                        lRatesConseq[i][j] = defLRate;
                }
            }


            for (int i = 0; i < z.Length; i++)
            {
                z_accum[i] = new double[outputDim];
                p_accum[i] = new double[terms[i].Parameters.Length];
            }
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

        private static double dEdP(double[] y, double[] o,
           double[][] z,
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
                        (grad[p] * (firingSum - firings[rule]) / (firingSum * firingSum)) :
                        (-firings[i] * grad[p] / (firingSum * firingSum))) * z[i][C];


                g += (o[C] - y[C]) * subSum;
            }
            if (Math.Abs(g) > 10)
                Console.WriteLine("");
            return g;
        }
    }
}
