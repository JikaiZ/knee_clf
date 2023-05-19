import torch
import torch.nn as nn
import torch.nn.functional as F

class BERT(nn.Module):

    def __init__(self, bert):
        super(BERT, self).__init__()
        self.encoder = bert

    def forward(self, text, mask, label):
        loss, logits = self.encoder(text, mask, labels=label).loss, self.encoder(text, mask, labels=label).logits
        text_fea = torch.sigmoid(logits)
        return loss, text_fea


class BERT_classifier(nn.Module):
	def __init__(self, bert, fc_dims, dropout_rate=0.1):
		super(BERT_classifier, self).__init__()
		self.encoder = bert
		self.dropout = nn.Dropout(dropout_rate)

		self.fc_size = len(fc_dims)
		self.fc = nn.ModuleList([nn.Linear(fc_dims[i], fc_dims[i+1]) for i in range(self.fc_size) if i <= self.fc_size-2])


	def forward(self, sent_id, mask):
		_, cls_hidden_state = self.encoder(sent_id, attention_mask=mask)
		X = self.dropout(cls_hidden_state)
		for i, layer in enumerate(self.fc):
			if i < self.fc_size - 2:
				X = F.relu(layer(F.dropout(X)))
			else:
				X = layer(X)
		output = X
		logits = F.softmax(X, dim=1)
		
		return output, logits

