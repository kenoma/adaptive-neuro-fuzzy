using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANFIS
{
    public interface ITerm
    {
        double Membership(double[] input);

        double[] Parameters { get; set; }
        double[] GetGradient(double[] point);
    }
}
