using ANFIS.misc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANFIS.membership
{
    public static class RuleSetFactory<T, G>
        where T : IRule, new()
        where G : IRuleExtractor, new()
    {
        /// <summary>
        /// Build initial ruleset from data
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <returns></returns>
        public static T[] Build(double[][] input, double[][] output, int RuleNumbers)
        {
            G rExtractor = new G();
            double[][] centroids;
            double[][] consequences;
            rExtractor.ExtractRules(input, output, RuleNumbers, out centroids, out consequences);

            T[] retVal = new T[centroids.Length];

            for (int c = 0; c < centroids.Length; c++)
            {
                retVal[c] = new T();
                int neigh = getNeighbourhood(centroids, c);
                retVal[c].Init(centroids[c], consequences[c], centroids[neigh]);
            }

            return retVal;
        }


        private static int getNeighbourhood(double[][] X, int c)
        {
            double mindis = double.MaxValue;
            int cand = c;
            for (int i = 0; i < X.Length; i++)
            {
                double d = math.EuclidianDistance(X[c], X[i]);
                if (i != c && d < mindis)
                {
                    cand = i;
                    mindis = d;
                }
            }
            return cand;
        }

    }
}
