from semantic_analysis import gloss_to_video
from semantic_analysis import word_vectors
from semantic_analysis import gloss_clusters
from semantic_analysis import vectors

import pandas as pd
import torch
import torch.nn as nn
import random
from english_asl import asl_sentences

#print(gloss_to_video)
#print(word_vectors)
#print(gloss_clusters)

#panda data-frame
df = pd.DataFrame(list(gloss_to_video.items()), columns=['video_id', 'gloss'])


gloss_to_cluster = {}
for cluster_id, cluster in enumerate(gloss_clusters):
    for gloss in cluster:
        gloss_to_cluster[gloss] = cluster_id

#add cluster to dataframe
df['cluster'] = df['gloss'].map(gloss_to_cluster)    

#if glosses are not in a cluster
df['cluster'] = df['cluster'].fillna(-1)


class Tokenizer:
    def __init__(self, glosses):
        self.gloss_to_id = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        for gloss in glosses:
            if gloss not in self.gloss_to_id:
                self.gloss_to_id[gloss] = len(self.gloss_to_id)
        self.id_to_gloss = {id: gloss for gloss, id in self.gloss_to_id.items()}
        
    
    def encode(self, sentence):
        return [self.gloss_to_id.get(word, self.gloss_to_id['<unk>']) for word in sentence.split()]
    
    def decode(self, ids):
        return ' '.join([self.id_to_gloss.get(id, '<unk>') for id in ids])




#tokenizer = Tokenizer(df['gloss'].unique())
#create tokenizer
all_glosses = set()
for src, trg in asl_sentences:
    all_glosses.update(src.split())
    all_glosses.update(trg.split())
tokenizer = Tokenizer(all_glosses)
print("Vocabulary size:", len(tokenizer.gloss_to_id))


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = torch.clamp(src, max=self.embedding.num_embeddings - 1)
        # if src.dim() == 1:
        #     src = src.unsqueeze(1) # (seq_len, 1)

        # embedded = self.dropout(self.embedding(src))

        # if embedded.dim() == 2:
        #     embedded = embedded.unsqueeze(1)        # (seq_len, 1, emb_dim)
            
        # outputs, (hidden, cell) = self.rnn(embedded)
        # return hidden, cell
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell):
        #input = input.view(1, -1)
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # #print("\ntrg shape:", trg.shape)
        # if src.dim() == 1:
        #     src = src.unsqueeze(1)
        # if trg.dim() == 1:  
        #     trg = trg.unsqueeze(1) #change shape to 2D
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim


        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)

        # Use the first token of the target sequence as input
        input = trg[0,:]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        
        return outputs


#Initialize Model
INPUT_DIM = OUTPUT_DIM = len(tokenizer.gloss_to_id)
ENC_EMB_DIM = 512
DEC_EMB_DIM = 512
HID_DIM = 1024
N_LAYERS = 3
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5


enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(enc, dec, device).to(device)

#from parallel_dataset import parallel_data


#prepare data
def prepare_data(sentences):
    src_tensors = []
    trg_tensors = []
    for src, trg in sentences:
        src_encoded = [tokenizer.gloss_to_id['<sos>']] + tokenizer.encode(src) + [tokenizer.gloss_to_id['<eos>']]
        trg_encoded = [tokenizer.gloss_to_id['<sos>']] + tokenizer.encode(trg) + [tokenizer.gloss_to_id['<eos>']]
        src_tensors.append(torch.tensor(src_encoded))
        trg_tensors.append(torch.tensor(trg_encoded))
    return src_tensors, trg_tensors
#src_tensors = [torch.tensor(tokenizer.encode(sent[0])) for sent in asl_sentences]
#trg_tensors = [torch.tensor(tokenizer.encode(sent[1])) for sent in asl_sentences]

src_tensors, trg_tensors = prepare_data(asl_sentences)
print("Sample src:", src_tensors[0])
print("Sample trg:", trg_tensors[0])

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence

class ASLDataset(Dataset):
    def __init__(self, src_tensors, trg_tensors):
        self.src_tensors = src_tensors
        self.trg_tensors = trg_tensors
    
    def __len__(self):
        return len(self.src_tensors)
    
    def __getitem__(self, idx):
        return self.src_tensors[idx], self.trg_tensors[idx]

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=tokenizer.gloss_to_id['<pad>'])
    trg_batch = pad_sequence(trg_batch, padding_value=tokenizer.gloss_to_id['<pad>'])
    return src_batch, trg_batch

# Create DataLoader
#dataset = TensorDataset(torch.stack(src_tensors), torch.stack(trg_tensors))
dataset = ASLDataset(src_tensors, trg_tensors)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


def train(model, src, trg, optimizer, criterion, clip):
    model.train()
    optimizer.zero_grad()
    output = model(src, trg)
    output_dim = output.shape[-1]
    output = output[1:].view(-1,  output_dim)
    trg = trg[1:].view(-1)
    loss = criterion(output, trg)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    return loss.item()

N_EPOCHS = 20
CLIP = 1

print("Vocabulary size:", model.encoder.embedding.num_embeddings)


def translate(sentence, model, tokenizer, device, max_length=50):
    model.eval()
    tokens = tokenizer.encode(sentence)
    tokens = [tokenizer.gloss_to_id['<sos>']] + tokens + [tokenizer.gloss_to_id['<eos>']]
    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

        trg_indexes = [tokenizer.gloss_to_id['<sos>']]

        for _ in range(max_length):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)

            if pred_token == tokenizer.gloss_to_id['<eos>']:
                break

    trg_tokens = tokenizer.decode(trg_indexes[1:-1]) #Remove <sos> and <eos>
    return trg_tokens

for epoch in range(N_EPOCHS):
    total_loss = 0
    model.train()
    for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            loss = train(model, src, trg, optimizer, criterion, CLIP)
            total_loss += loss
        
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}')

    # Evaluate on a sample sentence after each epoch
    model.eval()
    test_sentence = "I want to read a book and drink water using my computer"
    translation = translate(test_sentence, model, tokenizer, device)
    print(f"English: {test_sentence}")
    print(f"ASL Gloss: {translation}")
    print()

# Save the model after training
torch.save(model.state_dict(), 'asl_translation_model.pt')

#Verifying my Tokenizer
test_sentence = "I want to read a book and drink water using my computer"
encoded = tokenizer.encode(test_sentence)
decoded = tokenizer.decode(encoded)
print(f"Original: {test_sentence}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")


#Test
# test_sentence = "I want to read a book"
# translation = translate(test_sentence, model, tokenizer, device)
# print(f"English: {test_sentence}")
# print(f"ASL Gloss: {translation}")



