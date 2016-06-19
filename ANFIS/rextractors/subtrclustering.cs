using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuroFuzzy.misc;
using NLog;

namespace NeuroFuzzy.rextractors
{
    static class sbsclust
    {
        static object o = new object();
        static Logger _log = LogManager.GetLogger("sbsclust");
        static public double[][] SubstractiveClustering(double[][] x, double arad, double brad)
        {
            double[] P = new double[x.Length];
            arad = (arad / 2.0) * (arad / 2.0);
            brad = (brad / 2.0) * (brad / 2.0);
            int progress = 0;
            Parallel.For(0, x.Length, row =>
                {
                    for (int col = row + 1; col < x.Length; col++)
                    {
                        double dist = math.EuclidianDistance2(x[row], x[col]);
                        double add = Math.Exp(-dist / arad);
                        if(add!=0)
                            lock (o)
                            {
                                P[row] += add;
                                P[col] += add;
                            }
                    }
                    if (progress++ % 100 == 0)
                    {
                        _log.Info($"Potential {progress}/{x.Length}");
                    }
                });

            List<double[]> c = new List<double[]>();
            
            double D = P.Max();
            double cap = 0.01 * D;

            double[] wn = x[P.FindIndex(z => z == D)];
            c.Add(wn);
            while (D > cap)
            {
                _log.Info($"Iteration {c.Count} [{D} cap {cap}]");
                progress = 0;
                Parallel.For(0, x.Length, row =>
                {
                    double dist = math.EuclidianDistance2(x[row], wn);
                    double add = -D * Math.Exp(-dist / brad);
                    P[row] += add;

                    //if (progress++ % 100 == 0)
                    //{
                    //    Console.Write("\rPotential {0}/{1}", progress, x.Length);
                    //}
                });
                D = P.Max();
                wn = x[P.FindIndex(z => z == D)];
                c.Add(wn);
            }
            return c.ToArray();
        }

     
    }
}
