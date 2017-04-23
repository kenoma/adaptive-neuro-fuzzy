using CaseStudy.HeuristicsOne.FIS;
using NeuroFuzzy;
using NeuroFuzzy.membership;
using Newtonsoft.Json;
using OfficeOpenXml;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CaseStudy.HeuristicsOne
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.Write("Enter user data file [m5.csv]:");
            var fusers = Console.ReadLine();
            if (string.IsNullOrEmpty(fusers))
                fusers = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "m5.csv");

            if (!File.Exists(fusers))
            {
                Console.WriteLine($"No such file {fusers}");
                Console.ReadLine();
                Environment.Exit(0);
            }

            Console.Write("Enter m1 data [m1-pub.csv]:");
            var fm1 = Console.ReadLine();
            if (string.IsNullOrEmpty(fm1))
                fm1 = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "m1-pub.csv");

            if (!File.Exists(fm1))
            {
                Console.WriteLine($"No such file {fm1}");
                Console.ReadLine();
                Environment.Exit(0);
            }

            Console.Write("Enter m8 data [m8-vop.xlsx]:");
            var fm8 = Console.ReadLine();
            if (string.IsNullOrEmpty(fm8))
                fm8 = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "m8-vop.xlsx");

            if (!File.Exists(fm8))
            {
                Console.WriteLine($"No such file {fm8}");
                Console.ReadLine();
                Environment.Exit(0);
            }

            //s10;ID;URL;Название;Участников;s6;1;1.1;2;2.1;3;3.1;4;4.1;Статус;Screen_name;Описание;Страна;Город;Последний пост;Сайт;Тип сообщества (Открытое/Закрытое);Тип сообщества;Контакты;Фотографий;Фотоальбомов;Аудиозаписей;Видеозаписей;Обсуждений;Документов;Постов;Товаров;Место;Дата начала;Дата завершения;Количество возможных участников;Есть фото;Вики-страница;Можно постить на стену;Можно писать в сообщения;Можно просматривать все посты;Можно загружать видео;Ссылки;Официальное сообщество;Главная секция;Товары;Контакт в товарах;Возрастное ограничение
            var commonPublics = new List<CommonPublicData>();
            using (StreamReader file = new StreamReader(fm1))
            {
                var line = string.Empty;
                while ((line = file.ReadLine()) != null)
                {
                    var spl = line.Split(';');
                    if (spl.Length > 5 &&
                        int.TryParse(spl[1], out int groupId))
                    {
                        commonPublics.Add(new CommonPublicData
                        {
                            Id = groupId,
                            S6Code = spl[5]
                        });
                    }
                }
            }

            var em8 = new LinqToExcel.ExcelQueryFactory(fm8);

            var opposition = (from c in em8.Worksheet<SpecialPublicData>("Оппозиц#паблики")
                              select c).ToArray();
            var volunteer = (from c in em8.Worksheet<SpecialPublicData>("Волонт#паблики")
                             select c).ToArray();


            var userRawData = new List<UserData>();
            using (StreamReader file = new StreamReader(fusers))
            {
                var line = string.Empty;
                while ((line = file.ReadLine()) != null)
                {
                    var spl = line.Split('\t');
                    if (spl.Length > 2 &&
                        int.TryParse(spl[0], out int userId) &&
                        int.TryParse(spl[1], out int groupId))
                    {
                        userRawData.Add(new UserData
                        {
                            Id = userId,
                            GroupId = groupId
                        });
                    }
                }
            }

            var juxtaposition = new Dictionary<int, string>();
            foreach (var cp in commonPublics)
            {
                if (!juxtaposition.ContainsKey(cp.Id))
                    juxtaposition.Add(cp.Id, cp.S6Code);
                else
                    Console.WriteLine($"Duplicate public entry {cp.Id}-{cp.S6Code} | {cp.Id}-{juxtaposition[cp.Id]}");
            }
            foreach (var cp in opposition)
            {
                if (!juxtaposition.ContainsKey(cp.Id))
                    juxtaposition.Add(cp.Id, "s6-2");
                else
                    juxtaposition[cp.Id] = "s6-2";
            }
            foreach (var cp in volunteer)
            {
                if (!juxtaposition.ContainsKey(cp.Id))
                    juxtaposition.Add(cp.Id, "s6-3");
                else
                    juxtaposition[cp.Id] = "s6-3";
            }

            ReportCommon(userRawData, juxtaposition, opposition);
            ReportOne(userRawData, juxtaposition);
            ReportTwo(userRawData, juxtaposition);
            ReportThree(userRawData, opposition);
            
        }

        private static void ReportCommon(List<UserData> userRawData, Dictionary<int, string> juxtaposition, SpecialPublicData[] opposition)
        {
            var ruleset = new List<IRule>
            {
                new SimpleLinearRule(
                    new double[]{ 0.25, 0.01, 0.30, 0.0, 0.001, 0.00, 0, -1,0, 0, 0},
                    new double[]{ 0.80, 0.02, 0.80, 0.8, 0.020, 0.05, 3, 2 ,4, 5, 6},
                    1),
                new SimpleLinearRule(
                    new double[]{ 0.25, 0.5, 0.30, 0.0, 0.05, 0.00,1, -1, 2, 1, 1 },
                    new double[]{ 0.60, 0.1, 0.60, 0.6, 0.10, 0.05,6, 3 , 8, 8, 9},
                    2),
                new SimpleLinearRule(
                    new double[]{ 0.00, 0.00, 0.30, 0.0, 0.50, 0.00 ,1, -1, 6, 3, 7},
                    new double[]{ 0.40, 0.25, 0.60, 0.3, 0.30, 0.15,11, 4,11, 11, 11},
                    3),
                new SimpleLinearRule(
                    new double[]{ 0.00, 0.00, 0.00, 0.0, 0.10, 0.10 ,0, 0,0, 5, 0},
                    new double[]{ 0.30, 0.20, 0.30, 0.6, 0.40, 0.50,150, 150,7, 11, 11},
                    4),
                new SimpleLinearRule(
                    new double[]{ 0.00, 0.00, 0.00, 0.00, 0.10, 0.00 ,2, -1 ,2, 6, 5 },
                    new double[]{ 0.20, 0.20, 0.20, 0.15, 0.50, 0.30,150, 4 ,9, 11, 11},
                    5),
                new SimpleLinearRule(
                    new double[]{ 0.00, 0.00, 0.00, 0.0, 0.10, 0.00 ,2, -1,8, 2, 6},
                    new double[]{ 0.90, 0.20, 0.30, 0.4, 0.50, 0.05,21, 3,11, 9, 11},
                    6)
            };

            var fis = new ANFIS(ruleset);

            var groopy = userRawData.ToLookup(z => z.Id, z => z.GroupId).Where(z => z.Count() < 150);
            var sb = new StringBuilder();
            sb.AppendLine($"ID, {string.Join(", ", Enumerable.Range(0, 7).Select(z => $"Стадия {z}"))}");
            var trainset = new StringBuilder();
            trainset.AppendLine("ID, %s6-1a, %s6-1b, %s6-1c, %s6-1d, %s6-2, %s6-3, s6-2, s6-3, s7-1, s7-2, s7-3");

            var count = groopy.Count();
            var dic = opposition.ToDictionary(z => z.Id, z => z);
            foreach (var user in groopy)
            {
                var codes = user.Where(z => juxtaposition.ContainsKey(z)).Select(z => juxtaposition[z]);
                if (codes.Any())
                {
                    var overall = (double)codes.Count();
                    var line = $"{user.Key}, ";
                    var opp = new double[3];
                    foreach (var pub in user)
                        if (dic.ContainsKey(pub))
                        {
                            opp[0] = Math.Max(opp[0], dic[pub].s7_1);
                            opp[1] = Math.Max(opp[1], dic[pub].s7_2);
                            opp[2] = Math.Max(opp[2], dic[pub].s7_3);
                        }

                    var input = new double[]
                    {
                        codes.Count(z=>z.Equals("s6-1a"))/overall,
                        codes.Count(z=>z.Equals("s6-1b"))/overall,
                        codes.Count(z=>z.Equals("s6-1c"))/overall,
                        codes.Count(z=>z.Equals("s6-1d"))/overall,
                        codes.Count(z=>z.Equals("s6-2"))/overall,
                        codes.Count(z=>z.Equals("s6-3"))/overall,
                        codes.Count(z=>z.Equals("s6-2")),
                        codes.Count(z=>z.Equals("s6-3")),
                        opp[0],
                        opp[1],
                        opp[2]
                    };
                    var res = fis.Inference(input);
                    if (res.Any(z => double.IsNaN(z)))
                        res = new double[] { 1.0, 0, 0, 0, 0, 0, 0 };

                    line += string.Join(", ", res.Select(z => z.ToString("0.00", CultureInfo.InvariantCulture)));
                    sb.AppendLine(line);
                    trainset.AppendLine($"{user.Key},{string.Join(", ", input.Select(z => z.ToString("0.0000", CultureInfo.InvariantCulture)))}");
                }
                else
                    sb.AppendLine($"{user.Key}, {string.Join(", ", Enumerable.Range(0, 7).Select(z => "0.00"))}");

                Console.Write($"\r Remains {count--}             ");
            }
            File.WriteAllText("report_common.csv", sb.ToString());
            File.WriteAllText("origin.csv", trainset.ToString());
        }

        private static void ReportThree(List<UserData> userRawData, SpecialPublicData[] opposition)
        {
            var ruleset = new List<IRule>
            {
                new SimpleLinearRule(
                    new double[]{ 0, 0, 0 },
                    new double[]{ 4, 5, 6 },
                    1),
                new SimpleLinearRule(
                    new double[]{ 2, 1, 1 },
                    new double[]{ 8, 8, 9 },
                    2),
                new SimpleLinearRule(
                    new double[]{ 6, 3, 7 },
                    new double[]{ 11, 11, 11 },
                    3),
                new SimpleLinearRule(
                    new double[]{ 0, 5, 0 },
                    new double[]{ 7, 11, 11 },
                    4),
                new SimpleLinearRule(
                    new double[]{ 2, 6, 5 },
                    new double[]{ 9, 11, 11 },
                    5),
                new SimpleLinearRule(
                    new double[]{ 8, 2, 6 },
                    new double[]{ 11, 9, 11 },
                    6)
            };

            var fis = new ANFIS(ruleset);

            var groopy = userRawData.ToLookup(z => z.Id, z => z.GroupId).Where(z => z.Count() < 150);
            var sb = new StringBuilder();
            sb.AppendLine($"ID, {string.Join(", ", Enumerable.Range(0, 7).Select(z => $"Стадия {z}"))}");
            var count = groopy.Count();
            var dic = opposition.ToDictionary(z => z.Id, z => z);

            foreach (var user in groopy)
            {
                var line = $"{user.Key}, ";
                var input = new double[] { 0, 0, 0 };
                foreach (var pub in user)
                    if (dic.ContainsKey(pub))
                    {
                        input[0] = Math.Max(input[0], dic[pub].s7_1);
                        input[1] = Math.Max(input[1], dic[pub].s7_2);
                        input[2] = Math.Max(input[2], dic[pub].s7_3);
                    }

                var res = fis.Inference(input);
                if (res.Any(z => double.IsNaN(z)))
                    res = new double[] { 1.0, 0, 0, 0, 0, 0, 0 };

                line += string.Join(", ", res.Select(z => z.ToString("0.00", CultureInfo.InvariantCulture)));
                sb.AppendLine(line);

                Console.Write($"\r Remains {count--}             ");
            }
            File.WriteAllText("report_three.csv", sb.ToString());
        }

        private static void ReportOne(List<UserData> userRawData, Dictionary<int, string> juxtaposition)
        {
            var ruleset = new List<IRule>
            {
                new SimpleLinearRule(
                    new double[]{ 0.25, 0.01, 0.30, 0.0, 0.001, 0 },
                    new double[]{ 0.80, 0.02, 0.80, 0.8, 0.020, 0.05},
                    1),
                new SimpleLinearRule(
                    new double[]{ 0.25, 0.5, 0.30, 0.0, 0.05, 0 },
                    new double[]{ 0.60, 0.1, 0.60, 0.6, 0.10, 0.05},
                    2),
                new SimpleLinearRule(
                    new double[]{ 0.00, 0.00, 0.30, 0.0, 0.50, 0 },
                    new double[]{ 0.40, 0.25, 0.60, 0.3, 0.30, 0.15},
                    3),
                new SimpleLinearRule(
                    new double[]{ 0.00, 0.00, 0.00, 0.0, 0.10, 0.10 },
                    new double[]{ 0.30, 0.20, 0.30, 0.6, 0.40, 0.50},
                    4),
                new SimpleLinearRule(
                    new double[]{ 0.00, 0.00, 0.00, 0.00, 0.10, 0.00 },
                    new double[]{ 0.20, 0.20, 0.20, 0.15, 0.50, 0.30},
                    5),
                new SimpleLinearRule(
                    new double[]{ 0.00, 0.00, 0.00, 0.0, 0.10, 0.00 },
                    new double[]{ 0.90, 0.20, 0.30, 0.4, 0.50, 0.05},
                    6)
            };

            var fis = new ANFIS(ruleset);

            var groopy = userRawData.ToLookup(z => z.Id, z => z.GroupId).Where(z => z.Count() < 150);
            var sb = new StringBuilder();
            sb.AppendLine($"ID, {string.Join(", ", Enumerable.Range(0, 7).Select(z => $"Стадия {z}"))}");
            var count = groopy.Count();
            foreach (var user in groopy)
            {
                var codes = user.Where(z => juxtaposition.ContainsKey(z)).Select(z => juxtaposition[z]);
                if (codes.Any())
                {
                    var overall = (double)codes.Count();
                    var line = $"{user.Key}, ";
                    var input = new double[]
                    {
                        codes.Count(z=>z.Equals("s6-1a"))/overall,
                        codes.Count(z=>z.Equals("s6-1b"))/overall,
                        codes.Count(z=>z.Equals("s6-1c"))/overall,
                        codes.Count(z=>z.Equals("s6-1d"))/overall,
                        codes.Count(z=>z.Equals("s6-2"))/overall,
                        codes.Count(z=>z.Equals("s6-3"))/overall
                    };
                    var res = fis.Inference(input);
                    if (res.Any(z => double.IsNaN(z)))
                        res = new double[] { 1.0, 0, 0, 0, 0, 0, 0 };

                    line += string.Join(", ", res.Select(z => z.ToString("0.00", CultureInfo.InvariantCulture)));
                    sb.AppendLine(line);
                }
                else
                    sb.AppendLine($"{user.Key}, {string.Join(", ", Enumerable.Range(0, 7).Select(z => "0.00"))}");

                Console.Write($"\r Remains {count--}             ");
            }
            File.WriteAllText("report_one.csv", sb.ToString());
        }

        private static void ReportTwo(List<UserData> userRawData, Dictionary<int, string> juxtaposition)
        {
            var ruleset = new List<IRule>
            {
                new SimpleLinearRule(
                    new double[]{ -0.1, -1 },
                    new double[]{  0.1, 150 },
                    0),
                new SimpleLinearRule(
                    new double[]{  0, -1 },
                    new double[]{  3, 2 },
                    1),
                new SimpleLinearRule(
                    new double[]{  1, -1 },
                    new double[]{  6, 3 },
                    2),
                new SimpleLinearRule(
                    new double[]{  1, -1 },
                    new double[]{  11, 4 },
                    3),
                new SimpleLinearRule(
                    new double[]{  0, 0 },
                    new double[]{  150, 150 },
                    4),
                new SimpleLinearRule(
                    new double[]{  2, -1 },
                    new double[]{  150, 4 },
                    5),
                new SimpleLinearRule(
                    new double[]{  2, -1 },
                    new double[]{  21, 3 },
                    6),
            };

            var fis = new ANFIS(ruleset);

            var groopy = userRawData.ToLookup(z => z.Id, z => z.GroupId).Where(z => z.Count() < 150);
            var sb = new StringBuilder();
            sb.AppendLine($"ID, {string.Join(", ", Enumerable.Range(0, 7).Select(z => $"Стадия {z}"))}");
            var count = groopy.Count();
            foreach (var user in groopy)
            {
                var codes = user.Where(z => juxtaposition.ContainsKey(z)).Select(z => juxtaposition[z]);
                if (codes.Any())
                {
                    var line = $"{user.Key}, ";
                    var input = new double[]
                    {
                        codes.Count(z=>z.Equals("s6-2")),
                        codes.Count(z=>z.Equals("s6-3"))
                    };
                    var res = fis.Inference(input);
                    if (res.Any(z => double.IsNaN(z)))
                        res = new double[] { 1.0, 0, 0, 0, 0, 0, 0 };

                    line += string.Join(", ", res.Select(z => z.ToString("0.00", CultureInfo.InvariantCulture)));
                    sb.AppendLine(line);
                }
                else
                    sb.AppendLine($"{user.Key}, {string.Join(", ", Enumerable.Range(0, 7).Select(z => "0.00"))}");

                Console.Write($"\r Remains {count--}             ");
            }
            File.WriteAllText("report_two.csv", sb.ToString());
        }
    }
}
