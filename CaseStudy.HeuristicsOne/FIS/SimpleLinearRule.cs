using NeuroFuzzy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CaseStudy.HeuristicsOne.FIS
{
    class SimpleLinearRule : IRule
    {
        public double[] Z { get; set; }
        public double[] Parameters { get; set; }

        public double[] Centroid => throw new NotImplementedException();

        public double[] GetGradient(double[] point)
        {
            throw new NotImplementedException();
        }


        private double[] _low, _hight;

        public SimpleLinearRule(double[] LowBound, double[] HighBound, int res)
        {
            _low = LowBound;
            _hight = HighBound;
            Z = new double[7];
            Z[res] = 1.0;
            if (_low.Length != _hight.Length)
                throw new Exception();
        }

        public void Init(double[] LowBound, double[] Consequence, double[] HighBound)
        {
        }

        public double Membership(double[] input)
        {
            if (input.Length != _low.Length)
                throw new Exception();

            var res = 0.0;
            for (int i = 0; i < _low.Length; i++)
            {
                if (input[i] < _low[i] || input[i] > _hight[i])
                    continue;
                var a = _low[i];
                var b = 0.5 * (_low[i] + _hight[i]);
                var c = _hight[i];
                var x = input[i];

                res += Math.Max(0, Math.Min((x - a) / (b - a), (c - x) / (c - b)));
            }
            return res /= input.Length;
        }
    }
}
