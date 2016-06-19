using Newtonsoft.Json;
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
        /// <summary>
        /// Stores object to file with binary serialization
        /// </summary>
        /// <param name="pObject"></param>
        /// <param name="filename"></param>
        public static void Serialize(T pObject, string filename)
        {
            Stream stream = File.Open(filename, FileMode.Create);
            BinaryFormatter bformatter = new BinaryFormatter();
            bformatter.Serialize(stream, pObject);
            stream.Close();
        }

        /// <summary>
        /// Stores object to file with json serialization
        /// </summary>
        /// <param name="pObject"></param>
        /// <param name="filename"></param>
        public static void SerializeJSON(T pObject, string filename)
        {
            var json = JsonConvert.SerializeObject(pObject);
            File.WriteAllText(filename, json);
        }

        /// <summary>
        /// Read object from binary file
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public static T Deserialize(string filename)
        {
            T mp;
            Stream stream = File.Open(filename, FileMode.Open);
            BinaryFormatter bformatter = new BinaryFormatter();
            mp = (T)bformatter.Deserialize(stream);
            stream.Close();
            return mp;
        }

        /// <summary>
        /// Read object from json file
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public static T DeserializeJSON(string filename)
        {
            var json = File.ReadAllText(filename);
            var obj = JsonConvert.DeserializeObject<T>(json);
            return obj;
        }
    }

}
