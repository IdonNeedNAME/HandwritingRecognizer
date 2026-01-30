using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Security.RightsManagement;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Xml.Serialization;
using MathNet.Numerics.Distributions;
using myCNN;
using myReader;

namespace WpfApp1
{
    public partial class MainWindow : Window
    {
        public Trainer model;
        public String outputPath,state;
        public Data data;
        public DocReader tr,te;
        public int onetrain = 1500;//一次训练用的图
        public int cycles=50;//循环几次训练
        public int backcnt = 10, allcnt,onetest=5000;
        public bool stop = false;
        public MainWindow()
        {
            InitializeComponent();
            model = new Trainer();
            data =new Data(model,this);
            model.ini(null, true);
            model.buildDiffNet();
            tr = new DocReader();
            te = new DocReader();
            this.DataContext= data;
            allcnt = cycles * onetrain;
        }
        public void train(object sender, RoutedEventArgs e)
        {
            Task.Run(() =>
            {
                stop = false;
                tr.overrideRead(onetrain);
                double[] aver =new double[10];
                double maxx; int id=-1;
                int cnt = 0;
                for (int i = 0; i < cycles; i++)
                {
                    for (int j = 0; j < onetrain; j++,cnt++)
                    {
                        if (stop)
                        {
                            j--;cnt--;
                            Thread.Sleep(100);
                            continue;
                        }
                        maxx = 0;
                        model.genData(tr.pictures[j].pic);
                        for (int o = 0; o < 10; o++) { 
                            if (model.ansNode[o].value1 > maxx) { maxx= model.ansNode[o].value1; id = o; }
                            if (o == tr.pictures[j].ans) model.ansNode[o].backward(model.ansNode[o].value1 - 1);
                            else model.ansNode[o].backward(model.ansNode[o].value1);
                        }
                        model.recorder.uploadOneTest(tr.pictures[j].ans,id);
                        
                        if (cnt % backcnt == 0 && cnt != 0)
                        {
                            
                            data.answer=tr.pictures[j].ans;
                            data.myans = id;
                            for (int o = 0; o < 10; o++)
                            {
                                data.ansp[o]=model.ansNode[o].value1;  
                            }
                            data.finishRate = cnt *1.0f/allcnt ;
                            //if(data.finishRate < 0.8)
                            model.release(backcnt,Math.Sqrt(Math.Sqrt(Math.Sqrt(1-data.finishRate)))+0.00000001f);
                            //else
                            //model.release(backcnt, 0.001f);
                            data.OnPropertyChanged("state");
                            data.OnPropertyChanged("FinishRate");
                        }
                    }
                    
                }data.finishRate = 1;
                data.OnPropertyChanged("FinishRate");
            });
        }
        private void Button_Start(object sender, RoutedEventArgs e)
        {
            
        }

        public void test(object sender, RoutedEventArgs e)
        {
            Task.Run(() =>
            {
                data.acrecorder.cleanAll();
                stop = true;
                tr.overrideRead(onetest);
                for (int i = 0; i < onetest; i++) {
                    int ans = tr.pictures[i].ans, myans=-1;
                    model.genData(tr.pictures[i].pic);
                    double maxx = 0;
                    for (int j = 0; j < 10; j++) {
                        if (model.ansNode[j].value1 > maxx)
                        {
                            maxx= model.ansNode[j].value1;
                            myans = j;
                        }
                    }
                    data.acrecorder.uploadOneTest(ans, myans);
                    data.OnPropertyChanged("state");
                }
            });
        }
        public void refreshDoc(object sender, RoutedEventArgs e)
        {
            try
            {
                tr.path = data.TrainPath;
                tr.start();
            }
            catch {
                data.Warning = "文件读入失败1";
            }
            try
            {
                te.path = data.TestPath;
                te.start();
            }
            catch {
                data.Warning = "文件读入失败2";
            }
        }
        public void Stop(object sender, RoutedEventArgs e)
        {
            stop=true;
        }
        public void Donotstop(object sender, RoutedEventArgs e)
        {
            stop=false;
        }
    }
    public class Data : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        public void OnPropertyChanged( [CallerMemberName]  string propertyName=null) {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
        public Data(Trainer model, MainWindow win = null)
        {
            this.model = model;
            this.win = win;
            this.acrecorder = new TrainRecorder();
        }
        public string state
        {
            get
            {
                string ans;
                ans = "训练数据路径：";
                if (_trainpath ==null ||_trainpath.Length==0) ans += "no path\n"; else ans += TrainPath + "\n";
                ans += "测试数据路径：";
                if (_testpath == null || _testpath.Length == 0) ans += "no path\n"; else ans += TestPath + "\n";

                ans += "训练正确率:";
                if (model != null) ans+= (model.recorder.accuracy*100.0f).ToString("F3")+"%\n";
                else ans+= "no data\n";
                

                ans += "测试正确率:";
                if (model != null) ans += (acrecorder.accuracy * 100.0f).ToString("F3") + "%\n";
                else ans += "no data\n";

                ans += "测试数据：" + answer.ToString() + " 我的答案: " + myans.ToString()+"\n";
                for(int i = 0; i < 10; i++)
                {
                    ans += ansp[i].ToString("F3")+"  ";
                    if (i % 3 == 0) ans += "\n";
                }
                if (_warning != null && _warning.Length != 0) ans += _warning + "\n";
                return ans;
            }
        }
        public string FinishRate
        {
            get
            {
                return "finish: " + (100.0f*finishRate).ToString("F3")+"%";
            }
        }
        public string TrainPath
        {
            get { return _trainpath; }
            set { _trainpath = value; win.tr.path = value; OnPropertyChanged(); OnPropertyChanged("state"); }
        }
        public string TestPath
        {
            get { return _testpath; }
            set { _testpath = value; win.te.path = value; OnPropertyChanged(); OnPropertyChanged("state"); }
        }
        public string Warning
        {
            get { return _warning; }
            set { _warning = value; OnPropertyChanged("state"); }
        }
        public Trainer model;
        public double finishRate=0;
        public TrainRecorder acrecorder;
        string _trainpath= "C:\\Users\\19921\\Downloads\\digit-recognizer\\train.csv", _testpath = "C:\\Users\\19921\\Downloads\\digit-recognizer\\test.csv", _warning ;
        public int answer=-1, myans=-1;
        public double[] ansp=new double[10];
        public MainWindow win;
    }
   
}