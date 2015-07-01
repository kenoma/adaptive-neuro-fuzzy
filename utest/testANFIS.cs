using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANFIS.membership;
using System.Collections.Generic;
using ANFIS.training;
using System.Diagnostics;
using System.Linq;

namespace utest
{
    [TestClass]
    public class testANFIS
    {
        [TestMethod]
        public void TestGaussianMemb()
        {
            double[] centroid = new double[] { 0, 0, 0 };
            double[] scaling = new double[] { 1, 1, 1 };
            GaussianTerm gterm = new GaussianTerm(centroid, scaling);

            double[] x = new double[3];
            double memb = gterm.Membership(x);
            Assert.AreEqual(1, memb, 1e-10);

            x[0] = x[1] = x[2] = 1;

            memb = gterm.Membership(x);
            Assert.AreEqual(0.2231301601, memb, 1e-10);

            centroid[1] = 1;
            scaling[2] = 5;
            x[0] = 3;
            gterm = new GaussianTerm(centroid, scaling);
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
        public void TestBackprop1D()
        {
            BackpropTraining bprop = new BackpropTraining(5e-1,1e-3);

            ///will train x=y function
            int trainingSamples = 1000;
            double[][] x = new double[trainingSamples][];
            double[][] y = new double[trainingSamples][];
            double[][] tx = new double[trainingSamples][];
            double[][] ty = new double[trainingSamples][];
            
            Random rnd = new Random();

            for (int i = 0; i < trainingSamples; i++)
            {
                double val = rnd.NextDouble();
                x[i] = new double[] { val };
                y[i] = new double[] { val };

                val = rnd.NextDouble();
                tx[i] = new double[] { val };
                ty[i] = new double[] { val };
            }

            GaussianTerm[] terms = new GaussianTerm[] { 
                new GaussianTerm(new double[] { 0 }, new double[] { 1 }),
                new GaussianTerm(new double[] { 1 }, new double[] { 1 }) };
            double[][] z = new double[2][] { new double[1], new double[1] };

            double E = double.MaxValue;
            int epoch = 0;
            while (!bprop.isTrainingstoped())
            {  
                double trainingError = bprop.Iteration(x, y, z, terms);
                Trace.WriteLine(string.Format("Epoch {0} - Error {1}/{2}", epoch++, trainingError, bprop.Error(tx, ty, z, terms)), "training");
                Assert.IsTrue(E > trainingError, "growing descent");
                E = trainingError;
            }

            double[] tmp_x = new double[1];
            for (double vx = 0; vx < 1.0; vx += 1e-1)
            {
                tmp_x[0] = vx;
                double[] vy = ANFIS.ANFIS.Inference(tmp_x, z, terms, terms.Length, 1);
                Assert.AreEqual(vx, vy[0], 1e-2);
            }
        }

        [TestMethod]
        public void TestQprop1D()
        {
            QPropTraining qprop = new QPropTraining(5e-2);

            ///will train x=y function
            int trainingSamples = 1000;
            double[][] x = new double[trainingSamples][];
            double[][] y = new double[trainingSamples][];
            double[][] tx = new double[trainingSamples][];
            double[][] ty = new double[trainingSamples][];

            Random rnd = new Random();

            for (int i = 0; i < trainingSamples; i++)
            {
                double val = rnd.NextDouble();
                x[i] = new double[] { val };
                y[i] = new double[] { Math.PI * val };

                val = rnd.NextDouble();
                tx[i] = new double[] { val };
                ty[i] = new double[] { Math.PI * val };
            }

            GaussianTerm[] terms = new GaussianTerm[] { 
                new GaussianTerm(new double[] { 0 }, new double[] { 1 }),
                new GaussianTerm(new double[] { 1 }, new double[] { 1 }) };
            double[][] z = new double[2][] { new double[1], new double[1] };

            double E = double.MaxValue;
            int epoch = 0;
            while (!qprop.isTrainingstoped())
            {
                double trainingError = qprop.Iteration(x, y, z, terms);
                Trace.WriteLine(string.Format("Epoch {0} - Error {1}/{2}", epoch++, trainingError, qprop.Error(tx, ty, z, terms)), "training");
                //Assert.IsTrue(E > trainingError, "growing descent");
                E = trainingError;
            }

            double[] tmp_x = new double[1];
            for (double vx = 0; vx < 1.0; vx += 1e-1)
            {
                tmp_x[0] = vx;
                double[] vy = ANFIS.ANFIS.Inference(tmp_x, z, terms, terms.Length, 1);
                Assert.AreEqual(Math.PI * vx, vy[0], 3e-2);
            }
        }
              

        [TestMethod]
        public void TestBackprop2D()
        {
            BackpropTraining bprop = new BackpropTraining(1e-4, 1e-3);

            ///will train x=y function
            int trainingSamples = 1000;
            double[][] x = new double[trainingSamples][];
            double[][] y = new double[trainingSamples][];
            double[][] tx = new double[trainingSamples][];
            double[][] ty = new double[trainingSamples][];

            Random rnd = new Random();

            for (int i = 0; i < trainingSamples; i++)
            {
                bool isOrigin = i % 2 == 0;

                double[] c = new double[] { (isOrigin ? 0 : 1), (isOrigin ? 0 : 1) };

                double valx = c[0] + (isOrigin ? 0.4 : 0.8) * (0.5 - rnd.NextDouble());
                double valy = c[1] + (isOrigin ? 0.8 : 0.4) * (0.5 - rnd.NextDouble());

                double d = Math.Exp(-Math.Sqrt((valx - c[0]) * (valx - c[0]) + (valy - c[1]) * (valy - c[1])));

                x[i] = new double[] { valx, valy };
                y[i] = new double[] { isOrigin ? d : 0, isOrigin ? 0 : d };


                valx = c[0] + (isOrigin ? 0.4 : 0.8) * (0.5 - rnd.NextDouble());
                valy = c[1] + (isOrigin ? 0.8 : 0.4) * (0.5 - rnd.NextDouble());
                d = Math.Exp(-Math.Sqrt((valx - c[0]) * (valx - c[0]) + (valy - c[1]) * (valy - c[1])));

                tx[i] = new double[] { valx, valy };
                ty[i] = new double[] { isOrigin ? d : 0, isOrigin ? 0 : d };
            }

            GaussianTerm[] terms = new GaussianTerm[] { 
                new GaussianTerm(new double[] { 0.4,0.3 }, new double[] { 2e-1,1e-1 }),
                new GaussianTerm(new double[] { 0.6,0.6 }, new double[] { 1e-1,3e-1 }) };
            double[][] z = new double[2][] { new double[2] { 0, 0 }, new double[2] { 0, 0 } };

            double E = double.MaxValue;
            int epoch = 0;
            while (!bprop.isTrainingstoped())
            {
                double trainingError = bprop.Iteration(x, y, z, terms);
                Trace.WriteLine(string.Format("Epoch {0} - Error {1}/{2}", epoch++, trainingError, bprop.Error(tx, ty, z, terms)), "training");

                E = trainingError;
            }
            double testSetError = bprop.Error(tx, ty, z, terms);
            Assert.IsFalse(testSetError > 1e-2);
        }

        [TestMethod]
        public void TestBatchBackprop2D()
        {
            BatchBackpropTraining bprop = new BatchBackpropTraining(1e-3, 1e-3);

            ///will train x=y function
            int trainingSamples = 1000;
            double[][] x = new double[trainingSamples][];
            double[][] y = new double[trainingSamples][];
            double[][] tx = new double[trainingSamples][];
            double[][] ty = new double[trainingSamples][];

            Random rnd = new Random();

            for (int i = 0; i < trainingSamples; i++)
            {
                bool isOrigin = rnd.NextDouble() > 0.5;

                double valx = (isOrigin ? 0 : 1) + 0.5 - rnd.NextDouble();
                double valy = (isOrigin ? 0 : 1) + 0.5 - rnd.NextDouble();

                x[i] = new double[] { valx, valy };
                y[i] = new double[] { isOrigin ? 1 : 0, isOrigin ? 0 : 1 };

                isOrigin = rnd.NextDouble() > 0.5;
                valx = (isOrigin ? 0 : 1) + 0.5 - rnd.NextDouble();
                valy = (isOrigin ? 0 : 1) + 0.5 - rnd.NextDouble();

                tx[i] = new double[] { valx, valy };
                ty[i] = new double[] { isOrigin ? 1 : 0, isOrigin ? 0 : 1 };
            }

            GaussianTerm[] terms = new GaussianTerm[] { 
                new GaussianTerm(new double[] { 0.1,0.2 }, new double[] { 5e-1,5e-1 }),
                new GaussianTerm(new double[] { 0.6,1 }, new double[] { 5e-1,5e-1 }) };
            double[][] z = new double[2][] { new double[2] { 1, 0 }, new double[2] { 0, 1 } };

            double E = double.MaxValue;
            int epoch = 0;
            while (!bprop.isTrainingstoped())
            {
                double trainingError = bprop.Iteration(x, y, z, terms);
                Trace.WriteLine(string.Format("Epoch {0} - Error {1}/{2}", epoch++, trainingError, bprop.Error(tx, ty, z, terms)), "training");

                E = trainingError;
            }

            double testSetError = bprop.Error(tx, ty, z, terms);
            Assert.IsFalse(testSetError > 1e-2);
        }

        [TestMethod]
        public void TestQpropOptimization1()
        {
            QPropTraining qprop = new QPropTraining(1e-4);

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

            GaussianTerm[] terms = new GaussianTerm[] { 
                new GaussianTerm(new double[] { 0.5, 0.3 }, 
                    new double[] { 5e-1,5e-1 }) };

            double[][] z = new double[][] { new double[] { 0 } };

            string pp = "";

            foreach (var row in x.Take(100))
                pp += string.Format(System.Globalization.CultureInfo.InvariantCulture, "[{0}, {1}],", row[0], row[1]);

            double E = double.MaxValue;
            int epoch = 0;
            while (!qprop.isTrainingstoped())
            {
                double trainingError = qprop.Iteration(x, y, z, terms);
                Trace.WriteLine(string.Format("Epoch {0} - Error {1}/{2}", epoch++, trainingError, qprop.Error(tx, ty, z, terms)), "training");

                E = trainingError;
            }

            double testSetError = qprop.Error(tx, ty, z, terms);
            Assert.IsFalse(testSetError > 1e-2);
            Assert.AreEqual(z[0][0], 1.0, 1e-4);
        }

        [TestMethod]
        public void TestQpropOptimization2()
        {
            QPropTraining qprop = new QPropTraining(1e-4);

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

            GaussianTerm[] terms = new GaussianTerm[] { 
                new GaussianTerm(new double[] { 0.0 }, new double[] { 5e-1 }),
                new GaussianTerm(new double[] { 0.5 }, new double[] { 5e-1 }) };
            double[][] z = new double[2][] { new double[2] { 1, 0 }, new double[2] { 0, 1 } };

            double E = double.MaxValue;
            int epoch = 0;
            while (!qprop.isTrainingstoped())
            {
                double trainingError = qprop.Iteration(x, y, z, terms);
                Trace.WriteLine(string.Format("Epoch {0} - Error {1}/{2}", epoch++, trainingError, qprop.Error(tx, ty, z, terms)), "training");

                E = trainingError;
            }

            double testSetError = qprop.Error(tx, ty, z, terms);
            Assert.IsFalse(testSetError > 1e-2);
        }
    }
}
