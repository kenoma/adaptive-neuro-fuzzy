using NeuroFuzzy.membership;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroFuzzy
{
    public static class ABuilder<R>
        where R : IRule, new()
    {
        public static ANFIS Build(double[][] input, double[][] output, IRuleExtractor RuleExtractor, ITraining trainer, int MaxIterations)
        {
            Console.WriteLine("Start...");
            Console.WriteLine("Constructing initial rule set with [{0}]", RuleExtractor.GetType().Name);
            var ruleBase = RuleSetFactory<R>.Build(input, output, RuleExtractor).Select(z => z as IRule).ToList();
            Console.WriteLine("Get {0} initial rules.", ruleBase.Count);
            int epoch = 0;

            double trnError = 0.0;
            Console.WriteLine();
            Console.WriteLine();
            do
            {
                trnError = trainer.Iteration(input, output, ruleBase);
                Console.WriteLine("\t Epoch {0}, training error {1}", epoch, trnError);

                if (double.IsNaN(trnError))
                {
                    Console.WriteLine("Failure! Training error is NAN.");
                    throw new Exception("Failure! Bad system design.");
                }
            } while (!trainer.isTrainingstoped() && epoch++ < MaxIterations);

            
            ANFIS fis = new ANFIS(ruleBase);
            Console.WriteLine("Done");
            return fis;
        }
    }
}
