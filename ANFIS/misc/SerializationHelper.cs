using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace NeuroFuzzy
{
    public static class SerializationHelper<T>
    {
        public static void Serialize(T pObject, string filename)
        {
            Stream stream = File.Open(filename, FileMode.Create);
            BinaryFormatter bformatter = new BinaryFormatter();
            bformatter.Serialize(stream, pObject);
            stream.Close();
        }

        public static T Deserialize(string filename)
        {
            T mp;
            Stream stream = File.Open(filename, FileMode.Open);
            BinaryFormatter bformatter = new BinaryFormatter();
            mp = (T)bformatter.Deserialize(stream);
            stream.Close();
            return mp;
        }
    }

}
