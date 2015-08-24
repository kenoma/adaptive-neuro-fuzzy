using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroFuzzy
{
    /// <summary>
    /// Single rule for FIS with membership function parameters, consequences and rule itself
    /// </summary>
    public interface IRule
    {

        /// <summary>
        /// Consequence of the rule
        /// </summary>
        double[] Z { get; set; }

        /// <summary>
        /// Parameters of membership function flatterned to array
        /// </summary>
        double[] Parameters { get; set; }

        /// <summary>
        /// Centroid part of membership function. This is readonly property as centroid is stored in Parameters array
        /// </summary>
        double[] Centroid { get; }

        /// <summary>
        /// Return firing level of this rule
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        double Membership(double[] input);

        /// <summary>
        /// Get partial derivative of mf by parameters at point
        /// </summary>
        /// <param name="point"></param>
        /// <returns></returns>
        double[] GetGradient(double[] point);

        void Init(double[] Centroid, double[] Consequence, double[] NearestNeighb);

    }
}
