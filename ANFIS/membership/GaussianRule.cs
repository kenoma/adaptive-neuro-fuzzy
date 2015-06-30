using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANFIS.membership
{
    public class GaussianTerm : ITerm
    {
        int xdim;

        /// <summary>
        /// 0 :: xdim-1 = Centroid
        /// xdim :: 2*xdim-1 = Scaling
        /// </summary>
        double[] parameters;

        public GaussianTerm(double[] Centroid, double[] Scaling)
        {
            if (Centroid == null || Centroid.Length == 0 || Scaling == null || Scaling.Length != Centroid.Length)
                throw new Exception("Incorrect membership function initialization");

            xdim = Centroid.Length;
            parameters = new double[xdim * 2];
            Array.Copy(Centroid, parameters, xdim);
            Array.Copy(Scaling, 0, parameters, xdim, xdim);
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
