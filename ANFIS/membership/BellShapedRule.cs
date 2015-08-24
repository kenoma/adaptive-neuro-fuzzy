using NeuroFuzzy.misc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroFuzzy.membership
{
    /// <summary>
    /// Bell shaped membership function and its parameters and gradients
    /// mu = 1/(1+(||x-c||/a)^(2b))
    /// </summary>
    public class BellShapedRule : IRule
    {
        int xdim;

        double[] parameters;
        double[] centroid;
        double[] z;

        /// <summary>
        /// Initialization of bellshaped function
        /// </summary>
        /// <param name="Centroid">Centroid of memb function</param>
        /// <param name="Consequence">Consequence from rule</param>
        /// <param name="NearestNeighb">Nearest centroid of another rule</param>
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

        /// <summary>
        /// Parameters of bell shaped function, packed into array
        /// First n elements are rule centroid components and last two are 'a' and 'b' parameters resp.
        /// </summary>
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

        /// <summary>
        /// Centroid part of rule
        /// </summary>
        public double[] Centroid
        {
            get
            {
                Array.Copy(parameters, centroid, centroid.Length);
                return centroid;
            }
            private set { }
        }

        /// <summary>
        /// Consequence part of rule
        /// </summary>
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
        private double b { get { return parameters[xdim + 1]; } }

        /// <summary>
        /// mu = 1/(1+(||x-c||/a)^(2b))
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Membership(double[] x)
        {
            double sum = 0.0;
            for (int i = 0; i < xdim; i++)
                sum += pow2(x[i] - parameters[i]);
            sum /= pow2(a);
            sum = Math.Pow(sum, b);
            return 1.0 / (1.0 + sum);
        }

        /// <summary>
        /// Gradient for memb. function parameters
        /// </summary>
        /// <param name="point"></param>
        /// <returns></returns>
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
