using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuroFuzzy.misc;

namespace NeuroFuzzy.training
{
    public class StochasticBatch : ITraining
    {
        
        int RandomSampleSize = 100;
        double learningRate = 1e-10;
        double lastError = double.MaxValue;
        double abstol, reltol;
        bool isStop = false;
        BatchBackprop bprop;

        public StochasticBatch(int RandomSampleSize, double LearningRate, double abstol = 1e-5, double reltol = 1e-7, double adjustThreshold = 1e-15)
        {
            this.RandomSampleSize = RandomSampleSize;
            this.learningRate = LearningRate;
            this.abstol = abstol;
            this.reltol = reltol;
            
            bprop = new BatchBackprop(learningRate, abstol, reltol, adjustThreshold);
        }

        public double Iteration(double[][] x, double[][] y, IList<IRule> RuleBase)
        {
            isStop = false;
            if (!bprop.isAdjustingRules() && isAdjustingRules())
                bprop.UnknownCaseFaced += UnknownCaseFaced;

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

            double err = bprop.Iteration(subx, suby, RuleBase);
            //double globalerr = bprop.Error(x, y, RuleBase);

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
            return bprop.Error(x, y, RuleBase);
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
