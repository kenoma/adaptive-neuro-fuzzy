using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANFIS.membership;
using System.Collections.Generic;
using ANFIS.training;
using System.Diagnostics;
using System.Linq;
using ANFIS;
using ANFIS.rextractors;

namespace utest
{
    [TestClass]
    public class testANFIS
    {
        [TestMethod]
        public void TestGaussianMemb()
        {
            double[] dumb = new double[0];
            double[] centroid = new double[] { 0, 0, 0 };
            double[] scaling = new double[] { 1, 1, 1 };
            GaussianRule gterm = new GaussianRule();
            gterm.Init(centroid, dumb, centroid.Select((v, a) => 4.0 * (centroid[a] + scaling[a])).ToArray());

            double[] x = new double[3];
            double memb = gterm.Membership(x);
            Assert.AreEqual(1, memb, 1e-10);

            x[0] = x[1] = x[2] = 1;

            memb = gterm.Membership(x);
            Assert.AreEqual(0.2231301601, memb, 1e-10);

            centroid[1] = 1;
            scaling[2] = 5;
            x[0] = 3;
            gterm = new GaussianRule();
            gterm.Init(centroid, dumb, centroid.Select((v, a) => 4.0 * (centroid[a] + scaling[a])).ToArray());
            memb = gterm.Membership(x);
            Assert.AreEqual(0.01088902367, memb, 1e-10);

            double[] grad = gterm.GetGradient(x);

            Assert.AreEqual(0.03266707101, grad[0], 1e-10);
            Assert.AreEqual(0, grad[1], 1e-10);
            Assert.AreEqual(0.0004355609468, grad[2], 1e-10);

            Assert.AreEqual(0.09800121303, grad[3], 1e-10);
            Assert.AreEqual(0, grad[4], 1e-10);
            Assert.AreEqual(0.00008711218936, grad[5], 1e-10);
        }

        [TestMethod]
        public void TestOptimization1()
        {
            BackpropTraining bprop = new BackpropTraining(1e-2);
            BatchBackpropTraining bbprop = new BatchBackpropTraining(1e-2);
            QPropTraining qprop = new QPropTraining();

            int trainingSamples = 1000;
            double[][] x = new double[trainingSamples][];
            double[][] y = new double[trainingSamples][];
            double[][] tx = new double[trainingSamples][];
            double[][] ty = new double[trainingSamples][];

            Random rnd = new Random();

            for (int i = 0; i < trainingSamples; i++)
            {

                double valx = 0.5 - rnd.NextDouble();
                double valy = 0.5 - rnd.NextDouble();

                x[i] = new double[] { valx, valy };
                y[i] = new double[] { 1 };


                valx = 0.5 - rnd.NextDouble();
                valy = 0.5 - rnd.NextDouble();

                tx[i] = new double[] { valx, valy };
                ty[i] = new double[] { 1 };
            }

            subTestOptimization1(bprop, x, y, tx, ty);
            subTestOptimization1(bbprop, x, y, tx, ty);
            subTestOptimization1(qprop, x, y, tx, ty);
        }

        private static void subTestOptimization1(ITraining bprop, double[][] x, double[][] y, double[][] tx, double[][] ty)
        {
            GaussianRule[] terms = new GaussianRule[] { new GaussianRule()};
            terms[0].Init(
                new double[] { 0.5, 0.3 },
                new double[] { 0 },
                new double[] { 0.0, 0.0 });

            int epoch = 0;
            int maxit = 1000;
            double trnError = 0.0;
            double tstError = 0.0;

            do
            {
                trnError = bprop.Iteration(x, y, terms);
                tstError = bprop.Error(tx, ty, terms);
            } while (!bprop.isTrainingstoped() && epoch++ < maxit);

            Trace.WriteLine(string.Format("Epochs {0} - Error {1}/{2}", epoch, trnError, tstError), "training");
            Assert.IsFalse(tstError > 1e-2);
            Assert.IsFalse(trnError > 1e-2);
            Assert.AreEqual(terms[0].Z[0], 1.0, 1e-2);
        }

        [TestMethod]
        public void TestOptimization2()
        {
            BackpropTraining bprop = new BackpropTraining(1e-2);
            BatchBackpropTraining bbprop = new BatchBackpropTraining(1e-2);
            QPropTraining qprop = new QPropTraining();

            int trainingSamples = 100;
            double[][] x = new double[trainingSamples][];
            double[][] y = new double[trainingSamples][];
            double[][] tx = new double[trainingSamples][];
            double[][] ty = new double[trainingSamples][];

            Random rnd = new Random();

            for (int i = 0; i < trainingSamples; i++)
            {
                bool isRigth = i % 2 == 0;
                double valx = (isRigth ? 1 : -1) + (0.5 - rnd.NextDouble());

                x[i] = new double[] { valx };
                y[i] = new double[] { isRigth ? 1 : 0, isRigth ? 0 : 1 };

                valx = (isRigth ? 1 : -1) + (0.5 - rnd.NextDouble());

                tx[i] = new double[] { valx };
                ty[i] = new double[] { isRigth ? 1 : 0, isRigth ? 0 : 1 };
            }

            subTestOptimization2(bprop, x, y, tx, ty);
            subTestOptimization2(bbprop, x, y, tx, ty);
            subTestOptimization2(qprop, x, y, tx, ty);

        }

        private static void subTestOptimization2(ITraining bprop, double[][] x, double[][] y, double[][] tx, double[][] ty)
        {
            GaussianRule[] terms = new GaussianRule[] { 
                new GaussianRule(),
                new GaussianRule() };

            terms[0].Init(new double[] { 0.5 }, new double[2] { 1, 0 }, new double[] { 0.0 });
            terms[1].Init(new double[] { 0.0 }, new double[2] { 0, 1 }, new double[] { 0.5 });
            

            int epoch = 0;
            int maxit = 1000;
            double trnError = 0.0;
            double tstError = 0.0;

            do
            {
                trnError = bprop.Iteration(x, y, terms);
                tstError = bprop.Error(tx, ty, terms);
            } while (!bprop.isTrainingstoped() && epoch++ < maxit);

            Trace.WriteLine(string.Format("Epochs {0} - Error {1}/{2}", epoch, trnError, tstError), "training");
            Assert.IsFalse(tstError > 1e-2);
            Assert.IsFalse(trnError > 1e-2);
            Assert.AreEqual(terms[0].Z[0], 1.0, 1e-2);
            Assert.AreEqual(terms[0].Z[1], 0.0, 1e-2);
            Assert.AreEqual(terms[1].Z[0], 0.0, 1e-2);
            Assert.AreEqual(terms[1].Z[1], 1.0, 1e-2);
            Assert.IsTrue(terms[0].Parameters[0] > 0);
            Assert.IsTrue(terms[1].Parameters[0] < terms[0].Parameters[0]);
        }

        [TestMethod]
        public void TestRulesetGeneration()
        {
            int trainingSamples = 100;
            double[][] x = new double[trainingSamples][];
            double[][] y = new double[trainingSamples][];

            Random rnd = new Random();

            for (int i = 0; i < trainingSamples; i++)
            {
                bool isRigth = i % 2 == 0;
                double valx = (isRigth ? 1 : -1) + (0.5 - rnd.NextDouble());

                x[i] = new double[] { valx, valx };
                y[i] = new double[] { isRigth ? 1 : 0, isRigth ? 0 : 1 };
            }

            IRule[] ruleBase = RuleSetFactory<GaussianRule, KMEANSExtractor>.Build(x, y, 2);

            if (ruleBase[0].Z[0] > 0.5)
            {
                Assert.AreEqual(ruleBase[0].Z[0], 1, 1e-2);
                Assert.AreEqual(ruleBase[0].Z[1], 0, 1e-2);
                Assert.AreEqual(ruleBase[1].Z[1], 1, 1e-2);
                Assert.AreEqual(ruleBase[1].Z[0], 0, 1e-2);
                Assert.AreEqual(ruleBase[0].Parameters[0], 1, 1e-1);
                Assert.AreEqual(ruleBase[0].Parameters[1], 1, 1e-1);
                Assert.AreEqual(ruleBase[1].Parameters[0], -1, 1e-1);
                Assert.AreEqual(ruleBase[1].Parameters[1], -1, 1e-1);
            }
            else
            {
                Assert.AreEqual(ruleBase[0].Z[1], 1, 1e-2);
                Assert.AreEqual(ruleBase[0].Z[0], 0, 1e-2);
                Assert.AreEqual(ruleBase[1].Z[0], 1, 1e-2);
                Assert.AreEqual(ruleBase[1].Z[1], 0, 1e-2);
                Assert.AreEqual(ruleBase[1].Parameters[0], 1, 1e-1);
                Assert.AreEqual(ruleBase[1].Parameters[1], 1, 1e-1);
                Assert.AreEqual(ruleBase[0].Parameters[0], -1, 1e-1);
                Assert.AreEqual(ruleBase[0].Parameters[1], -1, 1e-1);
            }

        }

        [TestMethod]
        public void TestLogisticMap()
        {
            int trainingSamples = 1000;
            double[][] x = new double[trainingSamples][];
            double[][] y = new double[trainingSamples][];
            double[][] tx = new double[trainingSamples][];
            double[][] ty = new double[trainingSamples][];

            double px = 0.1;
            double r = 3.8;//3.56995;
            double lx = r * px * (1 - px);

            for (int i = 0; i < trainingSamples; i++)
            {
                x[i] = new double[] { px, lx };
                px = lx;
                lx = r * lx * (1 - lx);
                y[i] = new double[] { lx };
            }

            for (int i = 0; i < trainingSamples; i++)
            {
                tx[i] = new double[] { px, lx };
                px = lx;
                lx = r * lx * (1 - lx);
                ty[i] = new double[] { lx };
            }

            BackpropTraining bprop = new BackpropTraining(1e-2);
            BatchBackpropTraining bbprop = new BatchBackpropTraining(1e-2);
            QPropTraining qprop = new QPropTraining();

            subtestLogisticsMap(x, y, tx, ty, bprop);
            subtestLogisticsMap(x, y, tx, ty, bbprop);
            subtestLogisticsMap(x, y, tx, ty, qprop);
        }

        private static void subtestLogisticsMap(double[][] x, double[][] y, double[][] tx, double[][] ty, ITraining bprop)
        {
            ANFIS.ANFIS fis = ANFISFActory<GaussianRule, KMEANSExtractor>.Build(x, y, 5, bprop, 10000);
            double err = bprop.Error(tx, ty, fis.RuleBase);

            Trace.WriteLine(string.Format("[{1}] Logistic map Error {0}", err, bprop.GetType().Name), "training");
            Assert.IsFalse(err > 1e-1);
        }
    }
}
