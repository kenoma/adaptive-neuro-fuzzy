using NeuroFuzzy.misc;
using NLog;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroFuzzy.training
{
    public class QProp : ITraining
    {
        Logger _log;
        public event UnknownCase UnknownCaseFaced;

        double etaPlus, etaMinus,deltaMax,deltaMin,defLRate;
        double lastError = double.MaxValue;
        double abstol, reltol, adjustThreshold;

        bool isStop = false;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="abstol">stop when the function gets "close enough" to zero, i.e. smaller abstol </param>
        /// <param name="reltol">stop when the improvement drops below reltol</param>
        /// <param name="EtaPlus">Growing factor for descent step</param>
        /// <param name="EtaMinus">Shinking factor for descent step</param>
        /// <param name="DeltaMax">Max step limitation</param>
        /// <param name="DeltaMin">Min step limitation</param>
        /// <param name="InitialLearningRate">Default learning rate</param>
        public QProp(double abstol = 1e-5, double reltol = 1e-7, double EtaPlus = 1.2, double EtaMinus = 0.5, double DeltaMax = 1, double DeltaMin = 1e-8, double InitialLearningRate = 1e-4, double adjustThreshold = 1e-15)
        {
            _log = LogManager.GetLogger(this.GetType().Name);
            this.etaMinus = EtaMinus;
            this.etaPlus = EtaPlus;
            this.deltaMax = DeltaMax;
            this.deltaMin = DeltaMin;
            this.abstol = abstol;
            this.reltol = reltol;
            this.defLRate = InitialLearningRate;
            this.adjustThreshold = adjustThreshold;
        }


        private double[][] lRatesParams;
        private double[][] lRatesConseq;
        private double[][] prev_z_accum;
        private double[][] prev_p_accum;

        public double Iteration(double[][] x, double[][] y, IList<IRule> ruleBase)
        {
Restart:
            isStop=false;
            if (x.Length != y.Length)
                throw new Exception("Input and desired output lengths not match");
            if (ruleBase == null || ruleBase.Count == 0)
                throw new Exception("Incorrect rulebase");

            int outputDim = ruleBase[0].Z.Length;
            int numOfRules = ruleBase.Count;

            double[][] z_accum;
            double[][] p_accum;
            InitStuff(ruleBase, outputDim, out z_accum, out p_accum);

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

                if (UnknownCaseFaced != null && firingSum < adjustThreshold)
                {
                    int neig = math.NearestNeighbourhood(ruleBase.Select(z => z.Centroid).ToArray(), x[sample]);
                    UnknownCaseFaced(ruleBase, x[sample], y[sample], ruleBase[neig].Centroid);
                    _log.Info("Adjusting rule base. Now {0} are in base.", ruleBase.Count);
                    lRatesConseq = null;
                    lRatesParams = null;
                    prev_p_accum = null;
                    prev_z_accum = null;
                    goto Restart;
                }
          
                for (int i = 0; i < numOfRules; i++)
                    for (int C = 0; C < outputDim; C++)
                        o[C] += firings[i] / firingSum * ruleBase[i].Z[C];

                for (int rule = 0; rule < ruleBase.Count; rule++)
                {
                    //double[] parm = terms[rule].Parameters;
                    double[] grad = ruleBase[rule].GetGradient(x[sample]);

                    for (int p = 0; p < grad.Length; p++)
                    {
                        double g = dEdP(y[sample], o, ruleBase, firings, grad, firingSum, rule, outputDim, numOfRules, p);
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

            for (int rule = 0; rule < ruleBase.Count; rule++)
            {
                double[] parm = ruleBase[rule].Parameters;
                for (int p = 0; p < parm.Length; p++)
                    parm[p] -= lRatesParams[rule][p] * p_accum[rule][p];
            }

            for (int i = 0; i < numOfRules; i++)
                for (int C = 0; C < outputDim; C++)
                    ruleBase[i].Z[C] -= lRatesConseq[i][C] * z_accum[i][C];


            checkStop(globalError);

            
            return globalError / x.Length;
        }

        private void checkStop(double globalError)
        {
            if (globalError < abstol || Math.Abs(lastError - globalError) < reltol)
                isStop = true;

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

        private void InitStuff(IList<IRule> ruleBase, int outputDim, out double[][] z_accum, out double[][] p_accum)
        {
            z_accum = new double[ruleBase.Count][];
            p_accum = new double[ruleBase.Count][];

            if (lRatesParams == null)
            {
                lRatesParams = new double[p_accum.Length][];
                for (int i = 0; i < ruleBase.Count; i++)
                {
                    lRatesParams[i] = new double[ruleBase[i].Parameters.Length];
                    for (int j = 0; j < lRatesParams[i].Length; j++)
                        lRatesParams[i][j] = defLRate;
                }
            }
            if (lRatesConseq == null)
            {
                lRatesConseq = new double[z_accum.Length][];
                for (int i = 0; i < ruleBase.Count; i++)
                {
                    lRatesConseq[i] = new double[outputDim];
                    for (int j = 0; j < outputDim; j++)
                        lRatesConseq[i][j] = defLRate;
                }
            }


            for (int i = 0; i < ruleBase.Count; i++)
            {
                z_accum[i] = new double[outputDim];
                p_accum[i] = new double[ruleBase[i].Parameters.Length];
            }
        }

        public bool isTrainingstoped()
        {
            return isStop;
        }


        public double Error(double[][] x, double[][] y, IList<IRule> ruleBase)
        {
            if (x.Length != y.Length)
                throw new Exception("Input and desired output lengths not match");
            if (ruleBase == null || ruleBase.Count == 0)
                throw new Exception("Incorrect rulebase");

            int outputDim = ruleBase[0].Z.Length;
            int numOfRules = ruleBase.Count;

            double globalError = 0.0;

            for (int sample = 0; sample < x.Length; sample++)
            {
                double[] o = ANFIS.Inference(x[sample], ruleBase);
                for (int C = 0; C < outputDim; C++)
                    globalError += Math.Abs(o[C] - y[sample][C]);
            }

            return globalError / x.Length;
        }

    

        private static double dEdP(double[] y, double[] o,
           IList<IRule> z,
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
                        (-firings[i] * grad[p] / (firingSum * firingSum))) * z[i].Z[C];


                g += (o[C] - y[C]) * subSum;
            }
            if (Math.Abs(g) > 10)
                Console.WriteLine("");
            return g;
        }

        public bool isAdjustingRules()
        {
            return UnknownCaseFaced != null;
        }
    }
}
