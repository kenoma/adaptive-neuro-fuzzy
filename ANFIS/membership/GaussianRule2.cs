using NeuroFuzzy.misc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroFuzzy.membership
{
    public class GaussianRule2 : IRule
    {
        int xdim;

        /// <summary>
        /// 0 :: xdim-1 = Centroid
        /// xdim :: 2*xdim-1 = Scaling
        /// </summary>
        double[] parameters;
        double[] centroid;
        double[] z;

        public void Init(double[] Centroid, double[] Consequence, double[] NearestNeighb)
        {
            if (Centroid == null || Centroid.Length == 0 || Consequence == null)
                throw new Exception("Incorrect membership function initialization");

            xdim = Centroid.Length;
            parameters = new double[xdim + 1];
            centroid = new double[xdim];
            Array.Copy(Centroid, parameters, xdim);
            Array.Copy(Centroid, centroid, xdim);

            double desiredVatneigb = 1e-10;
            double d2 = math.EuclidianDistance2(Centroid, NearestNeighb);
            double a = Math.Sqrt(-d2 / (2 * Math.Log(desiredVatneigb)));
            parameters[xdim] = a;
            z = Consequence.ToArray();
        }

        public double[] Parameters
        {
            get
            {
                return parameters;
            }
            set
            {

                parameters = value;
            }
        }


        public double[] Centroid
        {
            get
            {
                Array.Copy(parameters, centroid, centroid.Length);
                return centroid;
            }
            private set { }
        }

        public double[] Z
        {
            get
            {
                return z;
            }
            set
            {
                z = value;
            }
        }

        /// <summary>
        /// mu = e^(- 0.5 ( ||x-c||/a )^2 )
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Membership(double[] x)
        {
            double exponent = 0.0;
            for (int i = 0; i < xdim; i++)
                exponent += pow2(x[i] - parameters[i]);

            double rval = Math.Exp(-0.5 * exponent / pow2(parameters[xdim]));

            return rval;
        }

        public double[] GetGradient(double[] point)
        {
            double[] grad = new double[xdim + 1];

            double exp = Membership(point);

            for (int i = 0; i < xdim; i++)
            {
                grad[i] = (point[i] - parameters[i]) * exp / pow2(parameters[xdim]);
                grad[xdim] += pow2(point[i] - parameters[i]);
            }
            grad[xdim] = exp * grad[xdim] / (parameters[xdim] * pow2(parameters[xdim]));
            return grad;
        }

        private double pow2(double x)
        {
            return x * x;
        }



    }
}
