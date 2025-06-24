# unet with FNO blocks

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels,out_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.relu = False
    def forward(self,x):
        return self.double_conv(x)


class FOUent(nn.Module):
    def __init__(self,in_channels,out_channels,features,modes=16): # choose of modes is up to us
        super(FOUent,self).__init__()
        
        self.encoder1 = DoubleConv(in_channels,features,features)
        self.encoder2 = DoubleConv(features,features*2,features*2)
        self.encoder3 = DoubleConv(features*2,features*4,features*4)
        self.encoder4 = DoubleConv(features*4,features*8,features*8)

        self.fo1 = FNOBlock(features*8,features*8,modes,modes)
        self.fo2 = FNOBlock(features*8,features*8,modes,modes)

        self.decoder4 = DoubleConv(features*16,features*4,features*4)
        self.decoder5 = DoubleConv(features*8,features*2,features*2)
        self.decoder6 = DoubleConv(features*4,features,features)

        self.final_conv = nn.Conv2d(features,out_channels,1)

        self.pool = nn.MaxPool2d(2)
        self.upconv4 = nn.ConvTranspose2d(features*8,features*8,kernel_size=2,stride=2)
        self.upconv3 = nn.ConvTranspose2d(features*4,features*4,kernel_size=2,stride=2)
        self.upconv2 = nn.ConvTranspose2d(features*2,features*2,kernel_size=2,stride=2)

    def forward(self,x):
         #encoder
     #   print(f"Input shape: {x.shape}")
        e1 = self.encoder1(x)  # number of channels is features
        # this converts the image into higher resolution we use conv here cause we need to extract features
      #  print(f"e1 shape: {e1.shape}")
        e2 = self.encoder2(self.pool(e1))  # number of channels is features *2
      #  print(f"e2 shape: {e2.shape}")
        
        e3 = self.encoder3(self.pool(e2)) # number of channels is features *4
      #  print(f"e3 shape: {e3.shape}")


        e4 = self.encoder4(self.pool(e3)) # number of channels is features * 8
      #  print(f"e4 shape: {e4.shape}")
        
        x = self.fo1(e4)
       # print(f"After fo1: {x.shape}")
        x = self.fo2(x)
       # print(f"After fo2: {x.shape}")

        # decoder
        x = self.upconv4(x) # the reason we ConvTranspose2d is we want to increase width and height number of channels is features * 8
        
        if x.size() != e4.size():
            x = torch.nn.functional.interpolate(x, size=e4.size()[2:], mode='bilinear', align_corners=False)
       # print(f"After upconv4: {x.shape}")

        x = torch.cat([x,e4],dim=1) # number of channels is features * 16
       # print(f"After cat with e4: {x.shape}")

        x = self.decoder4(x) # number of channels is features * 4
        #print(f"After decoder4: {x.shape}")


        x = self.upconv3(x) # number of channels is features * 4
       # print(f"After upconv3: {x.shape}")

        if x.size()[2:] != e3.size()[2:]:
            x = torch.nn.functional.interpolate(x, size=e3.size()[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x,e3],dim=1) #  number of channels is features * 8
       # print(f"After cat with e3: {x.shape}")

        x = self.decoder5(x) #  number of channels is features * 2
       # print(f"After decoder5: {x.shape}")


        x = self.upconv2(x) # number of channels is features * 2
        if x.size()[2:] != e2.size()[2:]:
            x = torch.nn.functional.interpolate(x, size=e2.size()[2:], mode='bilinear', align_corners=False)
        #print(f"After upconv2: {x.shape}")

        x = torch.cat([x,e2],dim=1) # number of channels is features * 4
       # print(f"After cat with e2: {x.shape}")

        x = self.decoder6(x) # number of channels is features 

        x = self.final_conv(x)
       # print(f"After decoder6: {x.shape}")
        x = torch.sigmoid(x)


        return x
