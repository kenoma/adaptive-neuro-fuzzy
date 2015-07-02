using ANFIS.misc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANFIS.membership
{
    public class GaussianRule : IRule
    {
        int xdim;

        /// <summary>
        /// 0 :: xdim-1 = Centroid
        /// xdim :: 2*xdim-1 = Scaling
        /// </summary>
        double[] parameters;
        double[] z;

        public void Init(double[] Centroid, double[] Consequence, double[] NearestNeighb)
        {
            if (Centroid == null || Centroid.Length == 0 || Consequence == null)
                throw new Exception("Incorrect membership function initialization");

            xdim = Centroid.Length;
            parameters = new double[xdim * 2];
            Array.Copy(Centroid, parameters, xdim);

            double[] gwidths = Centroid.Select((v, i) => 0.25 * Math.Max(1e-3, Math.Abs(Centroid[i] - NearestNeighb[i]))).ToArray();

            Array.Copy(gwidths, 0, parameters, xdim, xdim);
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
            double exponent = 0.0;
            for (int i = 0; i < xdim; i++)
                exponent += pow2((x[i] - parameters[i]) / parameters[i + xdim]);

            return Math.Exp(-0.5 * exponent);
        }

        public double[] GetGradient(double[] point)
        {
            double[] grad = new double[2 * xdim];

            double exp = Membership(point);

            for (int i = 0; i < xdim; i++)
            {

                grad[i] = (point[i] - parameters[i]) * exp / pow2(parameters[i + xdim]);
                grad[i + xdim] = pow2(point[i] - parameters[i]) * exp / (pow2(parameters[i + xdim]) * parameters[i + xdim]);
            }


            return grad;
        }

        private double pow2(double x)
        {
            return x * x;
        }



    }
}
