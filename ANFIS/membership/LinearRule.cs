using NeuroFuzzy.misc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroFuzzy.membership
{
    /// <summary>
    /// y=-||x-c||*a^(-1)+1,
    /// 
    /// y(0)=1
    /// y(a)=0
    /// </summary>
    public class LinearRule : IRule
    {
        int xdim;

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

            double d2 = math.EuclidianDistance2(Centroid, NearestNeighb);
            double a = Math.Sqrt(d2);

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

        private double a { get { return parameters[xdim]; } }
        
        /// <summary>
        /// mu = ||x-c||*a^(-1)+1,
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Membership(double[] x)
        {
            double sum = 0.0;
            for (int i = 0; i < xdim; i++)
                sum += pow2(x[i] - parameters[i]);
            sum = Math.Sqrt(sum);
            if (sum >= a)
                return 0.0;
            else
                return -sum / a + 1;
        }


        public double[] GetGradient(double[] point)
        {
            double[] grad = new double[xdim + 1];

            double sum = 0.0;
            for (int i = 0; i < xdim; i++)
                sum += pow2(point[i] - parameters[i]);
            sum = Math.Sqrt(sum);

            if (sum >= a)
            {
                for (int i = 0; i < xdim + 1; i++)
                    grad[i] = 0.0;
            }
            else
            {
                for (int i = 0; i < xdim; i++)
                    grad[i] = (point[i] - parameters[i]) / (a * sum);

                grad[xdim] = sum / pow2(a);
            }

            return grad;
        }

        private double pow2(double x)
        {
            return x * x;
        }



    }
}
