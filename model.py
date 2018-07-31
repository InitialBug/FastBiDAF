import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#--------------------------------------------------Base Module--------------------------------------------------#
class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model, num_updates):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                decay = min(self.mu, (1+num_updates)/(10+num_updates))
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]

# position embedding is direct computed
def PosEncoder(input, seq_len, dim):
    freqs = torch.Tensor([10000 ** (-i / dim) if i % 2 == 0 else -10000 ** (-(i - 1) / dim) for i in
                          range(dim)]).unsqueeze(dim=1)
    phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(dim)]).unsqueeze(dim=1)
    pos = torch.arange(seq_len).repeat(dim, 1)
    pos_embedding = torch.sin(torch.add(torch.mul(pos, freqs), phases))
    pos_embedding = pos_embedding.transpose(0, 1).to(device)
    out = input + pos_embedding
    return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_dim, out_dim, kernel_size, conv_dim=1, padding=0, bias=True, activation=False,dilation=1):
        super().__init__()
        self.activation = activation
        if conv_dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=input_dim, out_channels=out_dim, kernel_size=kernel_size,
                                            groups=input_dim,
                                            padding=padding, bias=bias,dilation=dilation)
            self.pointwise_conv = nn.Conv1d(in_channels=out_dim, out_channels=out_dim, kernel_size=1, padding=0,
                                            bias=bias)
        elif conv_dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=kernel_size,
                                            groups=input_dim,
                                            padding=padding, bias=bias,dilation=dilation)
            self.pointwise_conv = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1, padding=0,
                                            bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")

    def forward(self, input):
        input = input.transpose(1, 2)
        out = self.pointwise_conv(self.depthwise_conv(input))
        out = out.transpose(1, 2)
        if self.activation:
            return F.relu(out)
        else:
            return out


class Highway(nn.Module):
    def __init__(self, layer_num, dim):
        super().__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x

def GLU(input):
    out_dim=input.shape[2]//2
    a,b=torch.split(input,out_dim,dim=2)
    return a*F.sigmoid(b)

class DimReduce(nn.Module):
    def __init__(self, input_dim, out_dim,kernel_size):
        super().__init__()
        self.convout = nn.Conv1d(input_dim, out_dim*2, kernel_size, padding=kernel_size // 2)

    def forward(self, input):
        input = input.transpose(1, 2)
        input = self.convout(input)
        input = input.transpose(1, 2)
        out=GLU(input)
        return out

class Embedding(nn.Module):
    def __init__(self, char_dim, word_dim, word_len, out_dim, kernel_size, dropout=0.1):
        super().__init__()
        self.conv1d = DepthwiseSeparableConv(char_dim, char_dim, kernel_size, padding=kernel_size // 2,activation=True)
        self.highway = Highway(2, char_dim + word_dim)
        self.dropout = dropout
        self.char_dim = char_dim
        self.word_len = word_len

    def forward(self, word_emb, char_emb):
        batch_size = word_emb.shape[0]
        seq_len = word_emb.shape[1]
        char_emb = char_emb.view([-1, self.word_len, self.char_dim])
        char_emb = self.conv1d(char_emb)
        char_emb, _ = torch.max(char_emb, dim=1)
        char_emb = char_emb.view(batch_size, seq_len, self.char_dim)
        emb = torch.cat([word_emb, char_emb], dim=2)
        emb = self.highway(emb)
        return emb



class CQAttention(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.W = nn.Linear(input_dim * 3, 1)

    def forward(self, contex, question):
        contex_len = contex.shape[1]
        question_len = question.shape[1]
        c=contex.unsqueeze(dim=2)
        c=c.repeat([1,1,question_len,1])
        q=question.unsqueeze(dim=1)
        q=q.repeat([1,contex_len,1,1])
        S=torch.cat([q,c,q*c],dim=3)
        S=self.W(S).squeeze(dim=3)

        S1 = F.softmax(S, dim=2)
        S2 = F.softmax(S, dim=1)
        A = torch.bmm(S1, question)
        S3 = torch.bmm(S1, S2.transpose(1, 2))
        B = torch.bmm(S3, contex)
        out = torch.cat([contex, A, contex * A, contex * B], dim=2)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class Pointer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.W1 = nn.Linear(input_dim * 2, 1)
        self.W2 = nn.Linear(input_dim * 2, 1)

    def forward(self, M0, M1, M2):
        X1 = torch.cat([M0, M1], dim=2)
        X2 = torch.cat([M0, M2], dim=2)
        Y1 = self.W1(X1)
        Y2 = self.W2(X2)
        if self.training:
            # p1 = F.log_softmax(Y1, dim=1).squeeze(dim=2)
            # p2 = F.log_softmax(Y2, dim=1).squeeze(dim=2)
            p1 = Y1.squeeze(dim=2)
            p2 = Y2.squeeze(dim=2)
        else:
            p1 = F.softmax(Y1, dim=1).squeeze(dim=2)
            p2 = F.softmax(Y2, dim=1).squeeze(dim=2)
        return p1, p2



#--------------------------------------------------GLDR Module--------------------------------------------------#

class ResidualBlock(nn.Module):
    def __init__(self, input_dim,kernel_size,dilation):
        super().__init__()
        self.conv1=DepthwiseSeparableConv(input_dim,input_dim*2,kernel_size,padding=kernel_size//2*dilation,dilation=dilation)
        self.conv2=DepthwiseSeparableConv(input_dim,input_dim*2,kernel_size,padding=kernel_size//2*dilation,dilation=dilation)

    def forward(self, input):
        out=self.conv1(input)
        out=GLU(out)
        out=self.conv2(out)
        out=GLU(out)
        out=input+out
        return out

class GLDR(nn.Module):
    def __init__(self,input_dim,kernel_size,conv_num,exp_num=5,refine_num=3,dropout=0.1):
        super().__init__()
        self.dropout=dropout
        self.exp_conv=nn.Sequential()
        dilation=1
        for i in range(conv_num):
            self.exp_conv.add_module(str(i),ResidualBlock(input_dim,kernel_size,dilation))
            if i<exp_num:
                dilation*=2
        self.refine=nn.Sequential()
        for i in range(refine_num):
            self.refine.add_module(str(i),ResidualBlock(input_dim,kernel_size,dilation=1))


    def forward(self, input):
        out=self.exp_conv(input)
        out=self.refine(out)
        out=F.dropout(out,p=self.dropout,training=self.training)
        return out


class FastBiDAF(nn.Module):
    def __init__(self, char_dim, char_vocab_size, word_len, word_dim, word_mat, emb_dim, kernel_size,
                 encoder_block_num,model_block_num, dropout=0.1):
        super().__init__()
        self.char_emb = nn.Embedding(char_vocab_size, char_dim)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat))
        self.input_embedding = Embedding(char_dim, word_dim, word_len, emb_dim, kernel_size, dropout)
        self.dim_reduce1=DimReduce(char_dim+word_dim,emb_dim,kernel_size)
        self.embedding_encoder=GLDR(emb_dim,kernel_size,conv_num=encoder_block_num,dropout=dropout)
        self.cq_attention = CQAttention(emb_dim, dropout)
        self.dim_reduce2=DimReduce(emb_dim*4,emb_dim,kernel_size)
        self.model_encoder = GLDR(emb_dim,kernel_size,conv_num=model_block_num,dropout=dropout)
        self.pointer = Pointer(emb_dim)


    def forward(self, contex_word, contex_char, question_word, question_char):
        contex_word = self.word_emb(contex_word)
        contex_char = self.char_emb(contex_char)
        question_word = self.word_emb(question_word)
        question_char = self.char_emb(question_char)
        contex = self.input_embedding(contex_word, contex_char)
        contex=self.dim_reduce1(contex)
        question = self.input_embedding(question_word, question_char)
        question=self.dim_reduce1(question)
        contex = self.embedding_encoder(contex)
        question = self.embedding_encoder(question)
        CQ = self.cq_attention(contex, question)
        CQ=self.dim_reduce2(CQ)
        M0 = self.model_encoder(CQ)
        M1 = self.model_encoder(M0)
        M2 = self.model_encoder(M1)
        p1, p2 = self.pointer(M0, M1, M2)

        return p1, p2





# unit test
if __name__ == '__main__':
    device= torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # a=torch.randn([2,6,3],type=torch.Long)
    # b=torch.randn([2,6,5,3])
    # c=torch.randn([2,3,3])
    # d=torch.randn([2,3,5,3])
    # model1=DepthwiseSeparableConv(2,2,3,padding=3//2)
    # model2=Highway(2,2)
    # model3=SelfAttention(2,1,2)
    # model4=Embedding(3,3,5,2,3)
    # model5=EncoderBlock(2,2,3,2,2)
    # # a=model(a,b)
    # a=model4(a,b)
    # print(a.size())
    # c=model4(c,d)
    # print(c.shape)
    #
    # a=model5(a)
    # print(a.shape)
    # c=model5(c)

    # model6=CQAttention(3)
    # model7=Pointer(3,6)
    # p1,p2=model7(a,a,a)
    # print(p1.shape,p2.shape)
    # target = torch.empty(2, dtype=torch.long).random_(6)
    # print(target.shape)
    # exit()
    # ct=nn.CrossEntropyLoss()
    # loss=ct(p1,target)
    # print(loss)

    # modelseq=nn.Sequential()
    # for i in range(3):
    #     modelseq.add_module(str(i),nn.Linear(3,3))
    # a=modelseq(a)

    batch_size = 2
    para_len = 16
    question_len = 10
    word_dim = 3
    char_dim = 2
    word_len = 16
    word_vocab = 10
    char_vocab = 10
    kernel_size = 3
    att_map_dim = 2
    emb_dim = 4
    head_num = 3
    block_num = 3
    epoch = 100
    lr = 0.001

    contex_word = torch.empty([batch_size, para_len], dtype=torch.long).random_(word_vocab).to(device)
    contex_char = torch.empty([batch_size, para_len, word_len], dtype=torch.long).random_(char_vocab).to(device)
    question_word = torch.empty([batch_size, question_len], dtype=torch.long).random_(word_vocab).to(device)
    question_char = torch.empty([batch_size, question_len, word_len], dtype=torch.long).random_(char_vocab).to(device)
    word_mat = torch.randn([word_vocab, word_dim])
    model = FastBiDAF(char_dim, char_vocab, word_len, word_dim, word_mat, emb_dim, kernel_size,block_num,block_num).to(device)
    model.train()
    y1s = torch.empty(batch_size, dtype=torch.long).random_(para_len).to(device)
    y2s = torch.empty(batch_size, dtype=torch.long).random_(para_len).to(device)
    loss_func = nn.NLLLoss()
    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(betas=(0.8, 0.999), eps=1e-7, weight_decay=3e-7, params=parameters)
    crit = lr / math.log2(1000)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: crit * math.log2(
    #     ee + 1) if ee + 1 <= 1000 else lr)

    for i in range(epoch):
        optimizer.zero_grad()
        p1, p2 = model(contex_word, contex_char, question_word, question_char)
        loss1 = loss_func(p1, y1s)
        loss2 = loss_func(p2, y2s)
        loss = loss1 + loss2
        print(loss)
        loss.backward()
        optimizer.step()
        # scheduler.step()
    # a=torch.randn([2,15,4])
    # c=torch.randn([2,4,15])
    # # rs=ResidualBlock(4,3,2)
    # # b=rs(a)
    # d=6
    # conv=nn.Conv1d(4,4,3,padding=3//2*d,dilation=d)
    # b=conv(c)
    # print(b.shape)
