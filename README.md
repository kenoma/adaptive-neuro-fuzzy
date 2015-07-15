# .NET ANFIS
*This is WIP project*
###### About
This C# implementation of ANFIS (Adaptive Neuro Fuzzy Inference System) is designed to solve task *y=f(x)* in form of IF–THEN rules 
<blockquote><i>if x is A<sub>i</sub> then y is B<sub>i</sub></i></blockquote>
where <i>x</i> is an <i>m</i> dimensional input vector, and <i>y</i> is an <i>n</i> dimensional vector of desired output, <i>A<sub>i</sub></i>  is fuzzy set and <i>B<sub>i</sub></i> is consequence part of *i*-th rule.

Current version performs inference based on zero-order Sugeno fuzzy model (special case of the Mamdani Fuzzy inference system).

###### Algorithm

1. Perform clustering on datasets *x* and *y*, where *x* is an input dataset and *y* is a dataset of desired outputs . 
2. Initialize fuzzy sets *A<sub>i</sub>* and consequences *B<sub>i</sub>* with use of obtained clusters.
3. Tune ANFIS parameters with backprop in order to improve inference of the system.
  0. *(Optional)* if during training occurs situation when input case is not firing any rule, then it is possible to add new rule to database or adjust parameters of existing rules to fix issue.

###### Supported membership functions
1. Triangle
2. Bell-shaped
3. Gaussian

###### Example of usage
Following code generates training datasets of logistic map evolution in form *(x<sub>n-1</sub>, x<sub>n</sub>) → x<sub>n+1</sub>* and build ANFIS which can predict *x<sub>n+1</sub>* on two previouse values *(x<sub>n-1</sub>, x<sub>n</sub>)*.

```csharp
int trainingSamples = 2000;
double[][] x = new double[trainingSamples][];
double[][] y = new double[trainingSamples][];

double px = 0.1;
double r = 3.8;
double lx = r * px * (1 - px);

///generate training set
for (int i = 0; i < trainingSamples; i++)
{
    x[i] = new double[] { px, lx };
    px = lx;
    lx = r * lx * (1 - lx);
    y[i] = new double[] { lx };
}

///initialize trainig algorythm
Backprop bprop = new Backprop(1e-2);
///initialize clustering algo which will provide us initial parameters for rules
KMEANSExtractorIO extractor = new KMEANSExtractorIO(10);
///Build IS with Gaussian membershib functions
ANFIS.ANFIS fis = ANFISFActory<GaussianRule>.Build(x, y, extractor, bprop, 1000);
///[Backprop - GaussianRule] Error 0,000690883407351925	Elapsed 00:00:31.1691934	RuleBase 10
```
Now you can use trained `fis` as folowing
```csharp
double[] y = fis.Inference(x);
```
For more examles look to `testANFIS.cs`.