using NeuroFuzzy.misc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroFuzzy.membership
{
    public static class RuleSetFactory<T>
        where T : IRule, new()
    {
        /// <summary>
        /// Build initial ruleset from data
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <returns></returns>
        public static List<T> Build(double[][] input, double[][] output, IRuleExtractor RuleExtractor)
        {
            double[][] centroids;
            double[][] consequences;
            RuleExtractor.ExtractRules(input, output, out centroids, out consequences);

            List<T> retVal = new List<T>();

            for (int c = 0; c < centroids.Length; c++)
            {
                retVal.Add(new T());
                int neigh = math.NearestNeighbourhood(centroids, c);
                retVal[c].Init(centroids[c], consequences[c], centroids[neigh]);
            }

            return retVal;
        }


       

    }
}
