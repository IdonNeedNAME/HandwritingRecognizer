using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
namespace myReader
{
    public class DocReader
    {
        public string path;
        public List<Picture> pictures;
        public StreamReader reader;
        public int accu=0;
        public DocReader() {
             pictures = new List<Picture>();
             //reader = new StreamReader(path);
             //reader.ReadLine();
        }
        public void start()
        {
            reader = new StreamReader(path);
            reader.ReadLine();
        }
        public void overrideRead(int count)
        {
            Picture ptr;
            string content;
            for (int i = 1; i <= count; i++,accu++) {
                content = reader.ReadLine();
                if (string.IsNullOrWhiteSpace(content)) break;
                if (i > pictures.Count) {
                    ptr = new Picture();
                    ptr.pic = new double[28, 28];
                    pictures.Add(ptr);  
                }else ptr = pictures[i-1];

                string[] parts = content.Split(',');
                ptr.ans = int.Parse(parts[0]);
                for(int j = 1,xx=0,yy=0; j <= 784; j++)
                {
                    ptr.pic[xx,yy]=double.Parse(parts[j])/255.0f;

                    xx++;
                    if (j % 28 == 0) { xx = 0;yy++; }
                }
                

            }
        } 
    }
    public class Picture {
        public double[,] pic;
        public int ans;
        public Picture(double[,] pic,int ans) {
            this.ans = ans;
            this.pic = pic;
        }
        public Picture() { }
    }

}
