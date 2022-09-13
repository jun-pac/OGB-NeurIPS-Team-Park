class LinearModel(LightningModule):
    def __init__(self, model, input=768, hidden_layer=2048, output=153):
        super().__init__()
        self.l1 = Linear(input, hidden_layer)
        self.l2 = Linear(hidden_layer, output)
        self.dropout=Dropout(0.25)
        self.acc=Accuracy()
        self.acc_sum=0
        self.cnt=0
        self.model=model
    
    def forward(self, x):
        return torch.relu(self.l2(torch.relu(self.l1(x))))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat.softmax(dim=-1), y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        loss = F.cross_entropy(y_hat, y)
        if(batch_idx%100==0):
            print("Train accuracy : "+str(self.acc_sum/self.cnt)+"(Loss:"+str(loss.item())+") | Idx : "+str(batch_idx*x.shape[0])+" / 1112392 | Time : "+str(time.time()-t0))
            f_log.write("Train accuracy : "+str(self.acc_sum/self.cnt)+"(Loss:"+str(loss.item())+") | Idx : "+str(batch_idx*x.shape[0])+" / 1112392 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat.softmax(dim=-1), y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        if(batch_idx%100==0):
            print("Validation accuracy : "+str(self.acc_sum/self.cnt)+" | Idx : "+str(batch_idx*x.shape[0])+" / 138949 | Time : "+str(time.time()-t0))
            f_log.write("Validation accuracy : "+str(self.acc_sum/self.cnt)+" | Idx : "+str(batch_idx*x.shape[0])+" / 138949 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat.softmax(dim=-1), y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        if(batch_idx%100==0):
            print("Test accuracy : "+str(self.acc_sum/self.cnt)+" | Idx : "+str(batch_idx*x.shape[0])+" / 88092 | Time : "+str(time.time()-t0))
            f_log.write("Test accuracy : "+str(self.acc_sum/self.cnt)+" | Idx : "+str(batch_idx*x.shape[0])+" / 88092 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()

    def training_epoch_end(self, outputs) -> None:
        print("Epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write("Epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write('\n')
        f_log.flush()
        self.acc_sum=0
        self.cnt=0

    def validation_epoch_end(self, outputs) -> None:
        self.training_epoch_end(outputs)

    def test_epoch_end(self, outputs) -> None:
        self.training_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001) #0.001
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]