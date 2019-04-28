	package BPNet_test5;

	public class BPCoshaho
	{
	    // 第一层输入（原始输入）
	    private double[] x;
	    
	    // 第二层输入（隐含层结果）
	    private double[] y;
	    
	    // 第三层输入（预测结果）
	    private double[] z;
	    
	    // 实际结果
	    private double[] t;
	    
	    // 第一二层权重
	    private double[][] yx;
	    
	    // 第二三层权重
	    private double[][] zy;
	    
	    // 第三层（输出层）误差
	    private double[] zError;
	    
	    // 第二层（隐含层）误差
	    private double[] yError;
	    
	    // 学习速率
	    private double rate;
	    
	    /**
	     * 
	     * 正向计算输出值
	     *  input   上一层输出值
	     *  weight  上一层到本层的权重
	     *  output  本层输出值
	     */
	    private void calculateNextLevelValue(double[] input, double[][] weight, double[] output)
	    {
	        output[0] = 1d;
	        for(int i = 1; i < output.length; i++)
	        {
	            double temp = 0d;
	            // 下一层节点i的值为上一层每个节点与权重的乘积之和
	            for(int j = 0; j < input.length; j++)
	            {
	                temp = temp + weight[i - 1][j] * input[j];
	            }
	            // 数据归一化，使用1/(1+e^-x)函数
	            temp = 1d / (1d + Math.exp(-temp));
	            output[i] = temp;
	        }
	    }
	    
	    /**
	     * 
	     * 逆向计算误差
	     *  nextLevelError   下一层误差。计算输出层误差时，此参数为t
	     *  weight           本层到下一层权重。计算输出层误差时，w为1
	     *  output           本层输出值
	     *  error            本层误差
	     */
	    private void calculateError(double[] nextLevelError, double [][] weight, double[] output, double[] error)
	    {
	        // 输出层误差计算:每个输出层误差仅仅和一个目标值关联
	        if(null == weight)
	        {
	            for(int i = 0; i < nextLevelError.length; i++)
	            {
	                error[i] = (nextLevelError[i] - output[i + 1]) * output[i + 1] * (1 - output[i + 1]);
	            }
	            return;
	        }
	        
	        // 隐含层误差计算:每个隐含层误差对输出层所有误差都有影响
	        for(int i = 0; i < error.length; i++)
	        {
	            // 需要把输出层的所有误差分配到隐含层上
	            for(int j = 0; j < nextLevelError.length; j++)
	            {
	                // weight[j][i + 1]表上本层节点i到下一层节点j的权重
	                error[i] += nextLevelError[j] * weight[j][i + 1];
	            }
	            
	            // 误差经过权重累计后，还会使用激活函数进行计算，所以误差还需要乘以一个权重函数的导数
	            // f(x) = 1/(1+e^-x)导数为f(x)*(1-f(x))
	            error[i] = error[i] * output[i + 1] * (1 - output[i + 1]);
	        }
	    }
	    
	    /**
	     * 
	     * 更新权重
	     * error    本层误差
	     * weight   上一层到本层权重
	     * input    上一层输入
	     *  rate     学习速率
	     */
	    private void updateWeight(double[] error, double[][] weight, double[] input, double rate)
	    {
	        // 遍历权重列表调整权重
	        for(int i = 0; i < weight.length; i++)
	        {
	            for(int j = 0; j < weight[i].length; j++)
	            {
	                // weight[i][j]表示上一层节点j到本层节点i的权重
	                weight[i][j] += rate * error[i] * input[j];
	            }
	        }
	    }
	    
	    public BPCoshaho(int xSize, int ySize, int zSize, double rate)
	    {
	        if(xSize < 1 || ySize < 1 || zSize < 1 || 1.0d == rate)
	        {
	            throw new IllegalArgumentException("Parameter is error.");
	        }
	        
	        x = new double[xSize + 1];
	        y = new double[ySize + 1];
	        z = new double[zSize + 1];
	        x[0] = 1.0d;
	        y[0] = 1.0d;
	        z[0] = 1.0d;
	        
	        yx = new double[ySize][xSize + 1];
	        zy = new double[zSize][ySize + 1];
	        
	        zError = new double[zSize];
	        yError = new double[ySize];
	        
	        this.rate = rate;
	    }
	    
	    public void train(double[] x, double[] t)
	    {
	        // 输入训练数据
	        setX(x);
	        setT(t);
	        
	        // 正向传播计算输出值
	        calculateNextLevelValue(this.x, yx, y);
	        calculateNextLevelValue(y, zy, z);
	        
	        // 反向传播计算误差
	        calculateError(this.t, null, z, zError);
	        calculateError(zError, zy, y, yError);
	        
	        // 更新权重
	        updateWeight(yError, yx, this.x, rate);
	        updateWeight(zError, zy, this.y, rate);
	    }
	    
	    private void setX(double[] x)
	    {
	        if(null == x || x.length != this.x.length - 1)
	        {
	            throw new IllegalArgumentException("Input size is error.");
	        }
	        System.arraycopy(x, 0, this.x, 1, x.length);
	    }
	    
	    public double[] predict(double[] x)
	    {
	        double[] out = new double[z.length - 1];
	          setX(x);
	          
	        // 正向传播计算输出值
	        calculateNextLevelValue(this.x, yx, y);
	        calculateNextLevelValue(y, zy, z);
	        
	        System.arraycopy(z, 1, out, 0, out.length);
	        return out;
	    }
	    
	    private void setT(double[] t)
	    {
	        if(null == t || t.length != this.z.length - 1)
	        {
	            throw new IllegalArgumentException("Target size is error.");
	        }
	        this.t = t;
	    }
	}

<<<<<<< HEAD
=======
}
>>>>>>> branch 'master' of https://github.com/jianghuopro/test.git
