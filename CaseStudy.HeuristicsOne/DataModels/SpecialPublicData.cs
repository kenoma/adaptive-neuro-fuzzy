using LinqToExcel.Attributes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CaseStudy.HeuristicsOne
{
    class SpecialPublicData
    {
        [ExcelColumn("ID")]
        public int Id { get; set; }

        [ExcelColumn("s7-1 агрессия")]
        public double s7_1 { get; set; }

        [ExcelColumn("s7-2 разоблачения")]
        public double s7_2 { get; set; }

        [ExcelColumn("s7-3 фейк")]
        public double s7_3 { get; set; }
    }
}
