using System.Globalization;
using System.IO;
using MathNet.Numerics.Distributions;
namespace myCNN
{
    
    public abstract class DataNodeBase {
        public double learningRate,w,value1;
        public bool passable;
        public List<DataNodeBase> nexts;
        public virtual void backward(double error) { }
    }
    public class ConvolutionKernelNode : DataNodeBase {
        public ConvolutionKernelNode(double value,double learningRate,bool passable=true) {
            this.w = value;
            this.passable = passable;
            this.learningRate = learningRate;
            this.nexts = new List<DataNodeBase>();
        }
        public double accu=0;
        public override void backward(double error)
        {
            //w -= error * learningRate;
            accu += error * learningRate;
            foreach (DataNodeBase node in nexts)
            {
                node.backward(error);
            }
        }
    }
    public class SumNode : DataNodeBase
    {
        public SumNode()
        {
            this.nexts = new List<DataNodeBase>();
        }
        public List<double> diffs = new List<double>();//保存另一个乘数，在求导时多做一步传递回去 *已抛弃
        public bool shouldPassDiffs = false;//传递另一个乘数的开关
    }
    public class SimpleNode : DataNodeBase
    {
        public SimpleNode(double value, double learningRate, bool passable = true)
        {
            this.w = value;
            this.passable = passable;
            this.learningRate = learningRate;
            this.nexts = new List<DataNodeBase>();
        }
        public double accu=0;   
        public override void backward(double error)
        {
            //w-= error * learningRate*this.value1;    
            accu += error * this.value1*learningRate;
            foreach(DataNodeBase node in this.nexts)
            {
                node.backward(error*this.w);
            }
        }
    }
    public class TanhNode:DataNodeBase
    {
        public TanhNode()//使用tanh作为激活函数，导数是1-tanh^2
        {
            this.nexts=new List<DataNodeBase>();
        }
        public List<double> diffs = new List<double>();//保存另一个乘数，在求导时多做一步传递回去 
        public bool shouldPassDiffs = false;//传递另一个乘数的开关

        public override void backward(double error)
        {
            if (shouldPassDiffs)
            {
                for(int i = 0; i < nexts.Count; i++)
                {
                    nexts[i].backward(error * diffs[i]*(1-Math.Tanh(this.value1)* Math.Tanh(this.value1)));
                }
            }
            else
            {
                for( int i = 0;i < nexts.Count; i++)
                {
                    nexts[i].backward(error* (1 - Math.Tanh(this.value1) * Math.Tanh(this.value1)));
                }
            }
        }
    }
    public class SoftmaxNode : DataNodeBase
    {
        public SoftmaxNode() { this.nexts = new List<DataNodeBase>(); }
        public override void backward(double error) { 
            foreach(DataNodeBase next in nexts) next.backward(error);
        }
    }
    public class ConvolutionKernel
    {
        public List<ConvolutionKernelNode> nodes=new List<ConvolutionKernelNode>();
    }
    public class TrainRecorder {
        public List<int> answer, output;
        public int saveCount=10000,accnt=0;
        public double accuracy=0;
        public TrainRecorder()
        {
            answer = new List<int>();
            output = new List<int>();
        }
        public void uploadOneTest(int ans,int output)
        {
            answer.Add(ans);
            this.output.Add(output);
            if (ans == output) accnt++;
            if (answer.Count > saveCount)
            {
                if (answer[0] == this.output[0])
                {
                    accnt--;
                }
                answer.RemoveAt(0);
                this.output.RemoveAt(0);
            }
            if (answer.Count != 0)
                accuracy = accnt * 1.0 / answer.Count;
            else accuracy = 101;
        }
        public void cleanAll()
        {
            accnt = 0;
            answer.Clear();
            output.Clear();
        }
    }
    public class Trainer {
        public List <ConvolutionKernel> kernels;
        public List<SimpleNode> simpleNodes;//包含w参数和value1乘积
        public List<SoftmaxNode> ansNode; public int acnptr=0;
        public TrainRecorder recorder;
        public int kernelsCount=8,dimensions=10,outputWeightCount=1353;
        public double learningRate=0.01f;
        TanhNode[,,] futureMap=new TanhNode[8,13,13];
        SumNode[,] cacheMap = new SumNode[26, 26];
        public Trainer()//普通初始化
        {
            kernels= new List<ConvolutionKernel>();
            simpleNodes= new List<SimpleNode>();
            ansNode= new List<SoftmaxNode>();
            recorder = new TrainRecorder();
            for (int i = 0; i < dimensions; i++)//AnsNode ini
            {
                SoftmaxNode node=new SoftmaxNode();
                ansNode.Add(node);
            }
            for(int i=0;i<=25;i++) for(int j=0;j<=25;j++) cacheMap[i,j]=new SumNode();
            for (int i = 0; i < 8; i++) for (int j = 0; j < 13; j++) for (int k = 0; k < 13; k++)
            {
                futureMap[i, j, k] = new TanhNode();
                futureMap[i,j,k].shouldPassDiffs=true;
                        for (int kk = 0; kk <= 8; kk++) { futureMap[i, j, k].diffs.Add(0);  futureMap[i, j, k].nexts.Add(null); } 
            }
        }
        public void release(int times,double delta=1)
        {
            foreach(ConvolutionKernel kernel in kernels)
            {
                foreach(ConvolutionKernelNode node in kernel.nodes)
                {
                    node.w -= delta*node.accu / times;
                    node.accu = 0;
                }
            }
            foreach(SimpleNode node in simpleNodes)
            {
                node.w-=delta*node.accu / times;
                node.accu = 0;
            }
        }
        public void ini(string path,bool randomWeight)//初始化所有权重
        {
            Normal random = new Normal(0,0.165f*0.165f);
            Normal random2 = new Normal(0, 0.0135f * 0.0135f);
            simpleNodes.Clear();
            kernels.Clear();
            if (randomWeight) {//随机权重
                for(int i = 0; i < 8; i++) 
                {
                    kernels.Add(new ConvolutionKernel());
                    for(int j = 0; j < 9; j++)
                    {
                        kernels[i].nodes.Add(new ConvolutionKernelNode(random.Sample(), learningRate));
                    }
                }
                for(int i = 0; i < dimensions; i++)
                {
                    for(int j = 0; j < outputWeightCount-1; j++)
                    {
                        simpleNodes.Add(new SimpleNode(random2.Sample(), learningRate));
                    }
                    simpleNodes.Add(new SimpleNode(0,learningRate));
                }
            }
            else {//从路径中读取权重
            }
        }
        public void buildDiffNet()//连接误差传递网络
        {
            acnptr = 0;
            for (int i = 0; i < dimensions; i++) { 
                for(int j = 0; j < outputWeightCount; j++)
                {
                    ansNode[acnptr].nexts.Add(simpleNodes[i * outputWeightCount + j]);
                }
                acnptr++;
            }
        }
        public void genData(double[,] pic)//传入规定为28*28的图片
        {
            int ii = 0;
            foreach (ConvolutionKernel kernel in kernels)//池化和特征提取
            {
                double temp;
                for(int i = 0; i <= 25; i++)//--y
                {
                    for(int j=0; j <= 25; j++)//--x
                    {
                        //初始化cachemap，累计为0，传递乘数
                        cacheMap[j, i].value1 = 0;
                        cacheMap[j, i].shouldPassDiffs = true;
                        for (int k = 0,xx=j,yy=i; k <= 8; k++) {
                            //计算线性函数的一个项并累计并且记录乘数
                            temp = pic[xx, yy] * kernel.nodes[k].w;
                            cacheMap[j, i].value1 += temp;
                            if (k >= cacheMap[j, i].diffs.Count)
                                cacheMap[j, i].diffs.Add(pic[xx, yy]);
                            else cacheMap[j, i].diffs[k] = pic[xx,yy];

                            xx++;
                            if ((k + 1) % 3 == 0)
                            {
                                xx = j;
                                yy++;
                            }
                        }

                    }
                }
                //池化
                for (int i = 0; i <13; i++)//--y
                {
                    for (int j = 0; j <13; j++)//--x
                    {
                        futureMap[ii, j, i].value1 = cacheMap[j * 2, i * 2].value1;
                        for(int xx = j * 2, yy = i * 2, k = 0; k <= 3; k++)
                        {
                            if (cacheMap[xx, yy].value1 > futureMap[ii,j,i].value1)
                            {
                                futureMap[ii,j,i].value1 = cacheMap[xx, yy].value1;
                                for(int o = 0; o < 9; o++)//传递乘数
                                {
                                    futureMap[ii, j, i].diffs[o]=cacheMap[xx, yy].diffs[o];
                                }
                            }

                            xx++;
                            if (k % 2 == 1)
                            {
                                yy++;
                                xx = j * 2;
                            }
                        }
                        futureMap[ii, j, i].value1 = Math.Tanh(futureMap[ii,j,i].value1);
                        for(int jj = 0; jj < kernel.nodes.Count; jj++)
                        {
                            futureMap[ii, j, i].nexts[jj] = kernel.nodes[jj];
                        }
                    }
                }
                ii++;
            }
            double fm = 0;
            for(int i = 0,ind=0,cnt; i < dimensions; i++,ind++)//计算最终结果
            {
                double temp;
                ansNode[i].value1 = 0;cnt = 0;
                foreach(TanhNode node in futureMap)//枚举特征node
                {
                    temp = simpleNodes[ind].w * node.value1;
                    simpleNodes[ind].value1 = node.value1;
                    if (simpleNodes[ind].nexts.Count == 0)
                        simpleNodes[ind].nexts.Add(node);
                    else simpleNodes[ind].nexts[0] = node;
                    ansNode[i].value1 += temp;
                    //乘数保存在simpleNode中
                    //ansNode[i].diffs[cnt]=node.value1; 
                    cnt++;
                    ind++;
                }
                ansNode[i].value1 += simpleNodes[ind].w;
                simpleNodes[ind].value1 =1;
                fm += Math.Exp(ansNode[i].value1);
            }
            for (int i = 0; i < dimensions; i++)
            {
                ansNode[i].value1=Math.Exp(ansNode[i].value1)/fm;
            }
        }
    }

}
