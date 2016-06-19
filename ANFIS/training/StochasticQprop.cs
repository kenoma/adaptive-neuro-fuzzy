using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuroFuzzy.misc;
using NLog;

namespace NeuroFuzzy.training
{
    public class StochasticQprop : ITraining
    {
        int RandomSampleSize = 100;

        
        double lastError = double.MaxValue;
        double abstol, reltol;
        bool isStop = false;
        QProp qprop;

        public StochasticQprop(int RandomSampleSize, double abstol = 1e-5, double reltol = 1e-7, double EtaPlus = 1.2, double EtaMinus = 0.5, double DeltaMax = 1, double DeltaMin = 1e-8, double InitialLearningRate = 1e-4, double adjustThreshold = 1e-15)
        {
            this.RandomSampleSize = RandomSampleSize;
            this.abstol = abstol;
            this.reltol = reltol;

            qprop = new QProp(abstol, reltol, EtaPlus, EtaMinus, DeltaMax, DeltaMin, InitialLearningRate, adjustThreshold);
        }

        public double Iteration(double[][] x, double[][] y, IList<IRule> RuleBase)
        {
            isStop = false;
            if (!qprop.isAdjustingRules() && isAdjustingRules())
                qprop.UnknownCaseFaced += UnknownCaseFaced;

            var seq = x.Select((z, i) => i).ToArray();
            seq.Shuffle();

            double[][] subx = new double[RandomSampleSize][];
            double[][] suby = new double[RandomSampleSize][];

            int count = Math.Min(seq.Length, RandomSampleSize);
            for (int i = 0; i < count; i++)
            {
                subx[i] = x[seq[i]];
                suby[i] = y[seq[i]];
            }

            double err = qprop.Iteration(subx, suby, RuleBase);
            checkStop(err);
            lastError = err;
            return err;
        }

        private void checkStop(double globalError)
        {
            if (globalError < abstol)
                isStop = true;

            if (Math.Abs(lastError - globalError) < reltol)
                isStop = true;

            lastError = globalError;
        }

        public double Error(double[][] x, double[][] y, IList<IRule> RuleBase)
        {
            return qprop.Error(x, y, RuleBase);
        }

        public event UnknownCase UnknownCaseFaced;

        public bool isTrainingstoped()
        {
            return isStop;
        }


        public bool isAdjustingRules()
        {
            return UnknownCaseFaced != null;
        }
    }
}
