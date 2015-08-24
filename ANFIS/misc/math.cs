using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroFuzzy.misc
{
    public static class math
    {
        /// <summary>
        /// Get index for item in an IEnumerable
        /// Be careful as not all IEnumerables guarantees order preservation
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list"></param>
        /// <param name="finder"></param>
        /// <returns></returns>
       public static int FindIndex<T>(this IEnumerable<T> list, Predicate<T> finder)
        {
            int index = 0;
            foreach (var item in list)
            {
                if (finder(item))
                {
                    return index;
                }
                index++;
            }
            return -1;
        }

        /// <summary>
        /// Randomly shuffles an sequence
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list"></param>
        public static void Shuffle<T>(this IList<T> list)
        {
            Random rng = new Random(Environment.TickCount);
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        /// <summary>
        /// Returns index of closest to input element of centers
        /// </summary>
        /// <param name="inp"></param>
        /// <param name="centers"></param>
        /// <returns></returns>
        static public int DetectBucket(double[] inp, double[,] centers)
        {
            double minsum = double.MaxValue;
            int candidat = -1;
            for (int c = 0; c < centers.GetLength(0); c++)
            {
                double sum = 0;
                for (int i = 0; i < centers.GetLength(1); i++)
                {
                    double tmp = inp[i] - centers[c, i];
                    sum += tmp * tmp;
                    if (sum > minsum) break;
                }
                if (sum < minsum)
                {
                    minsum = sum;
                    candidat = c;
                }
            }
            return candidat;
        }

        static public double EuclidianDistance2(double[] inp, double[] c)
        {
            double dist = 0.0;
            for (int i = 0; i < c.Length; i++)
            {
                double x = inp[i] - c[i];
                dist += x * x;
            }
            return dist;
        }

        static public double EuclidianDistance(double[] inp, double[] c)
        {
            double dist = 0.0;
            for (int i = 0; i < c.Length; i++)
            {
                double x = inp[i] - c[i];
                dist += x * x;
            }
            return Math.Sqrt(dist);
        }

        static public double EuclidianDistance(int x, int y, double[,] c)
        {
            double dist = 0.0;
            for (int i = 0; i < c.GetLength(1); i++)
            {
                double tmp = c[x, i] - c[y, i];
                dist += tmp * tmp;
            }
            return Math.Sqrt(dist);
        }

        static public double EuclidianDistance(int x, int y, double[][] c)
        {
            double dist = 0.0;
            for (int i = 0; i < c[0].Length; i++)
            {
                double tmp = c[x][i] - c[y][i];
                dist += tmp * tmp;
            }
            return Math.Sqrt(dist);
        }


        static public int DetectBucket(double[] inp, double[][] centers)
        {
            double minsum = double.MaxValue;
            int candidat = -1;
            for (int c = 0; c < centers.GetLength(0); c++)
            {
                double sum = 0;
                for (int i = 0; i < centers[0].Length; i++)
                {
                    double tmp = inp[i] - centers[c][i];
                    sum += tmp * tmp;
                    if (sum > minsum) break;
                }
                if (sum < minsum)
                {
                    minsum = sum;
                    candidat = c;
                }
            }
            return candidat;
        }

        public static int NearestNeighbourhood(double[][] X, int c)
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

        public static int NearestNeighbourhood(double[][] X, double[] c)
        {
            double mindis = double.MaxValue;
            int cand = -1;
            for (int i = 0; i < X.Length; i++)
            {
                double d = math.EuclidianDistance(c, X[i]);
                if ( d < mindis)
                {
                    cand = i;
                    mindis = d;
                }
            }
            return cand;
        }
    }

}
