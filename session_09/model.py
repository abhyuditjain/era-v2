import torch.nn as nn


class DepthwiseConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super(DepthwiseConv, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=padding,
                groups=in_channels,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 0,
        use_depthwise: bool = False,
        use_skip: bool = False,
        dilation: int = 1,
        dropout: float = 0,
    ) -> None:
        super(ConvLayer, self).__init__()

        self.use_skip = use_skip

        if use_depthwise and in_channels == out_channels:
            self.conv_layer = DepthwiseConv(
                in_channels=in_channels,
                out_channels=out_channels,
                padding=padding,
                dilation=dilation,
            )
        else:
            self.conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False,
            )

        self.norm_layer = nn.BatchNorm2d(out_channels)

        self.skip_layer = None

        if use_skip and in_channels != out_channels:
            self.skip_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            )

        self.activation_layer = nn.ReLU()

        self.drop_layer = None
        if dropout > 0:
            self.drop_layer = nn.Dropout(dropout)

    def forward(self, x):
        y = x
        x = self.conv_layer(x)
        x = self.norm_layer(x)

        if self.use_skip:
            x += y if self.skip_layer is None else self.skip_layer(y)

        x = self.activation_layer(x)

        if self.drop_layer is not None:
            x = self.drop_layer(x)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        padding: int = 0,
        use_depthwise: bool = False,
        use_skip: bool = False,
        dilation: int = 1,
        dropout: float = 0,
    ):
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(
            *[
                ConvLayer(
                    in_channels=(in_channels if i == 0 else out_channels),
                    out_channels=out_channels,
                    padding=padding,
                    use_depthwise=use_depthwise,
                    use_skip=use_skip,
                    dilation=dilation,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class TransBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 0,
        use_depthwise: bool = False,
        use_skip: bool = False,
        dilation: int = 1,
        dropout: float = 0,
    ):
        super(TransBlock, self).__init__()

        self.trans_block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=padding,
            use_depthwise=use_depthwise,
            use_skip=use_skip,
            dilation=dilation,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.trans_block(x)
        return x


class Session09MModel(nn.Module):
    def __init__(
        self,
        dropout: float = 0,
        use_skip: bool = False,
    ):
        super(Session09MModel, self).__init__()

        self.dropout = dropout

        self.conv_block1 = ConvBlock(
            in_channels=3,
            out_channels=25,
            padding=1,
            use_depthwise=True,
            use_skip=False,
            num_layers=2,
            dropout=self.dropout,
        )
        self.trans_block1 = TransBlock(
            in_channels=25,
            out_channels=32,
            padding=0,
            use_depthwise=False,
            use_skip=False,
            dilation=1,
            dropout=self.dropout,
        )
        self.conv_block2 = ConvBlock(
            in_channels=32,
            out_channels=32,
            padding=1,
            use_depthwise=True,
            use_skip=use_skip,
            num_layers=2,
            dropout=self.dropout,
        )
        self.trans_block2 = TransBlock(
            in_channels=32,
            out_channels=65,
            padding=0,
            use_depthwise=False,
            use_skip=False,
            dilation=2,
            dropout=self.dropout,
        )
        self.conv_block3 = ConvBlock(
            in_channels=65,
            out_channels=65,
            padding=1,
            use_depthwise=True,
            use_skip=use_skip,
            num_layers=2,
            dropout=self.dropout,
        )
        self.trans_block3 = TransBlock(
            in_channels=65,
            out_channels=95,
            padding=0,
            use_depthwise=False,
            use_skip=False,
            dilation=4,
            dropout=self.dropout,
        )
        self.conv_block4 = ConvBlock(
            in_channels=95,
            out_channels=95,
            padding=1,
            use_depthwise=True,
            use_skip=use_skip,
            num_layers=2,
            dropout=self.dropout,
        )
        self.trans_block4 = TransBlock(
            in_channels=95,
            out_channels=95,
            padding=0,
            use_depthwise=False,
            use_skip=False,
            dilation=8,
            dropout=0,
        )

        self.out_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(95, 10, 1, bias=True),
            nn.Flatten(),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.trans_block1(x)
        x = self.conv_block2(x)
        x = self.trans_block2(x)
        x = self.conv_block3(x)
        x = self.trans_block3(x)
        x = self.conv_block4(x)
        x = self.trans_block4(x)
        x = self.out_block(x)

        return x
