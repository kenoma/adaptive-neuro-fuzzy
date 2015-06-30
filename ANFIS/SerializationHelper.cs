using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace ANFIS
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

        //public static void SerializefzProtoBuf(T pObject, string filename)
        //{
        //    using (System.IO.FileStream fs = new System.IO.FileStream(filename, System.IO.FileMode.Create))
        //        ProtoBuf.Serializer.Serialize(fs, pObject);
        //}

        //public static T DeserializefzProtobuf(string filename)
        //{
        //    T tr;
        //    using (System.IO.FileStream fs = new System.IO.FileStream(filename, System.IO.FileMode.Open))
        //        tr = ProtoBuf.Serializer.Deserialize<T>(fs);
        //    return tr;
        //}
    }

}
