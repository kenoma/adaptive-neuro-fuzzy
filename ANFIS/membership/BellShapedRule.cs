using ANFIS.misc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANFIS.membership
{
    public class BellShapedRule : IRule
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
            parameters = new double[xdim + 2];
            centroid = new double[xdim];
            Array.Copy(Centroid, parameters, xdim);
            Array.Copy(Centroid, centroid, xdim);

            double small =1e-10;
            double d2 = math.EuclidianDistance2(Centroid, NearestNeighb);
            double a = Math.Sqrt(d2) / 8;

            parameters[xdim] = a;
            parameters[xdim + 1] = Math.Log((1 - small) / small) / Math.Log(d2 / (a * a));
            
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
            set { }
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
        /// mu = e^(- 0.5 ( (x-c)/a )^2 )
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Membership(double[] x)
        {
            double sum = 0.0;
            for (int i = 0; i < xdim; i++)
                sum += pow2(x[i] - parameters[i]);
            sum /= pow2(parameters[xdim]);
            sum = Math.Pow(sum, parameters[xdim + 1]);
            return 1.0 / (1.0 + sum);
        }

        private double a { get { return parameters[xdim]; } }
        private double b { get { return parameters[xdim + 1]; } }

        public double[] GetGradient(double[] point)
        {
            double[] grad = new double[xdim + 2];

            double sum = 0.0;
            for (int i = 0; i < xdim; i++)
                sum += pow2(point[i] - parameters[i]);
            sum /= pow2(a);
            double tmp = sum;
            sum = Math.Pow(sum, b);

            for (int i = 0; i < xdim; i++)
                grad[i] = 2.0 * sum * b * (point[i] - parameters[i]) / (pow2(1 + sum) * tmp * pow2(a));

            grad[xdim] = 2.0 * sum * b / (pow2(1 + sum) * a);
            grad[xdim + 1] = -sum * Math.Log(tmp) / pow2(1 + sum);

            return grad;
        }

        private double pow2(double x)
        {
            return x * x;
        }



    }
}
