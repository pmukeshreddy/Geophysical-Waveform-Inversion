class FNOBlock(nn.Module):
    def __init__(self,in_channels,out_channels,modes1,modes2,width=32):
        super(FNOBlock,self).__init__()
        self.conv = SpectralConv2d(in_channels,out_channels,modes1,modes2)
        self.w = nn.Conv2d(in_channels,out_channels,1) # it is evivalent to linear project
        self.activation = nn.GELU()
    def forward(self,x):
        return self.activation(self.conv(x)+self.w(x)) # this propse is resudual connection
