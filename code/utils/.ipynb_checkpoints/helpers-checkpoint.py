import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import os, datetime, torch
from torch.utils.data import Dataset, DataLoader

def get_label_score_dict(row):
    '''
    Converts output of zero shot classification into a dictionary
    '''
    result_dict = dict()
    for _label, _score in zip(row['labels'], row['scores']):
        result_dict.update({'sequence': row['sequence']})
        result_dict.update({_label: _score})
    return result_dict

def find_thresholds(pred_prob_df, y_true, group_labels):
    '''
    pred_prob_df = dataframe of probabilities. rows = sequences, columns = labels. Extra columns can be present
    y_true = dataframe of the handlabelled data. Extra columns can be present
    group_labels = grouped labels
    '''
    
    thresholds = np.arange(0.1, 1, 0.05).round(2)
    label_thresholds = {c: 0 for c in group_labels}
    
    for col in group_labels:
        f1_list = []
        precision_list = []
        recall_list = []
    
        for threshold in thresholds:    

            preds = pred_prob_df[col].apply(lambda x: 1 if x >= threshold else 0)
            f1_list.append(f1_score(y_true=y_true[col], y_pred=preds))
            precision_list.append(precision_score(y_true=y_true[col], y_pred=preds))
            recall_list.append(recall_score(y_true=y_true[col], y_pred=preds))
        
        plt.figure()
        plt.plot(thresholds, f1_list, label='F1', marker='o', ms=3)
        plt.plot(thresholds, precision_list, label='Precision', marker='o', ms=3)
        plt.plot(thresholds, recall_list, label='Recall', marker='o', ms=3)
        plt.legend(bbox_to_anchor=(1, 1))
        plt.title(col)
        
        label_thresholds[col] = thresholds[np.array(f1_list).argmax()] # to return
    
    pred_df = pred_prob_df.copy()
    for col in group_labels:
        pred_df[col] = pred_df[col].apply(lambda x: 1 if x >= label_thresholds[col] else 0) # to return
        
    print(classification_report(y_true[group_labels].values, pred_df[group_labels].values, target_names=y_true[group_labels].columns, zero_division=0))
    
    return (label_thresholds, pred_df)

def get_scores(model, X_train, y_train, X_test, y_test):
    '''
    X_train = tfidf-vectorized training set
    X_test = tfidf-vectorized test set
    '''
    
    model.fit(X_train, y_train)
    pred_y_train = model.predict(X_train)
    pred_y_test = model.predict(X_test)
    
    print(f'------------TRAINING DATASET RESULTS ({model})------------')
    print('\n')
    print(classification_report(y_train.values, pred_y_train, target_names=y_train.columns, zero_division=0))
    print(f'------------TEST DATASET RESULTS ({model})------------')
    print('\n')
    print(classification_report(y_test.values, pred_y_test, target_names=y_test.columns, zero_division=0))


def augment_data(df, augmenter, column, n):
    
    df_col = df[df[column] == 1]
    
    augmented_rows_list = []
    
    for i in range(len(df_col)):
        
        print(f'Processing row {i+1} out of {len(df_col)} now...')
        row = df_col.iloc[[i]]
        row_text = row['sequence'].iloc[0]
        row_labels = row.drop(columns=['sequence'])

        text_added = pd.Series(augmenter.augment(row_text, n=n), name='sequence')
        labels_added = pd.DataFrame(np.repeat(row_labels.values, repeats=n, axis=0), columns=row_labels.columns)

        augmented_rows = pd.concat([text_added, labels_added], axis=1)
        augmented_rows_list.append(augmented_rows)
        
    print(f'Rows added: {n * len(df_col)}')
    
    rows_added = pd.concat(augmented_rows_list).reset_index(drop=True).drop_duplicates(subset=['sequence'])
    
    #final_df = pd.concat([df, rows_added]).reset_index(drop=True).drop_duplicates(subset=['sequence'])
    
    print(f'Shape of resulting dataframe: {rows_added.shape}')
    
#     final_df[['communication_behavior', 'waiting time_response time',
#        'information_knowledge', 'network', 'interface', 'facilities_parking',
#        'location', 'navigation_search', 'appointment', 'price']].sum().sort_values().plot(kind='barh')
    
    return rows_added

def preprocess_w2v(df,
                   w2v,
                    lower_case=True,
                    remove_special_punc=True):
    
    w2v_vocab = set(sorted(w2v.index_to_key))
       
    if lower_case:
        df['sequence'] = df['sequence'].str.lower() #lower case
    
    if remove_special_punc:
        df['sequence'] = df['sequence'].str.replace('[^0-9a-zA-Z\s]','') #remove special char, punctuation

    # Remove OOV words
    df['sequence_clean'] = df['sequence'].apply(lambda x: ' '.join([i for i in x.split() if i in w2v_vocab]))

    # Remove rows with blank string
    df.dropna(inplace=True)
    df = df[df['sequence_clean'] != ''].reset_index(drop=True)
    print(f'Shape of text: {df.shape}')
    
    # Vectorise text and store in new dataframe. Sentence vector = average of word vectors
    series_text_vectors = pd.DataFrame(df['sequence_clean'].apply(lambda x: np.mean([w2v[i] for i in x.split()], axis=0)).values.tolist())
    print(f'Shape of vectors: {series_text_vectors.shape}')
    
    return df.join(series_text_vectors)

def get_sentiment_label_twitter(list_of_sent_dicts):
    if list_of_sent_dicts[0]['score'] >= 0.5:
        return 0
    else:
        return 1
    
def get_sentiment_label_facebook(list_of_sent_dicts):
    if list_of_sent_dicts['labels'][0] == 'negative':
        return 0
    else:
        return 1
    
def get_sentiment_label_sst(list_of_sent_dicts):
    arg_max = np.array([i['score'] for i in list_of_sent_dicts]).argmax()
    return arg_max

class CustomTextDataset(Dataset):
    def __init__(self, text, tokenizer, max_len):
        self.text = text
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        inputs = self.tokenizer(
            text=self.text[idx],
            return_tensors='pt',
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'ids': ids,
            'mask': mask
        }
    
def preprocess_bert(model, tokenizer, dataframe, batch_size=2):
    '''
    Takes in a dataframe including the text, performs vectorization followed by train_test_split
    '''
    # Select columns
    df_bert = dataframe['sequence'].tolist()
    
    # Create Dataset object
    bert_dataset = CustomTextDataset(text = df_bert, 
                                 tokenizer = tokenizer,
                                 max_len = 512
                                )
    
    # Load Dataset into DataLoader
    bert_dataloader = DataLoader(bert_dataset, batch_size=batch_size)
    
    # Initialize empty array to store model output batches
    output_array = np.empty((0,768))
    
    # Iterate over batches
    for (idx, batch) in enumerate(bert_dataloader):
        
        # Print only if multiple of 100 (approx loading time ~1 hr)
        if idx % 100 == 0:
            print(f'Batch {idx} started at {datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}')
        
        # Obtain token ids. Squeeze to shape array to feed model
        ids = batch['ids'].view(-1,512)

        # Obtain attention masks. Squeeze to shape array to feed model
        mask = batch['mask'].view(-1,512)

        # no_grad = disable gradient calculation because we are only performing predictions
        with torch.no_grad():
            # Use [CLS] token as sentence vector, cast to numpy array
            output = model(ids, mask).last_hidden_state[:,0,:].numpy()
            output_array = np.vstack((output_array, output))
    
    print(f'Shape of bert vectors: {output_array.shape}')
    
    # Convert to DataFrame
    df_output_array = pd.DataFrame(output_array)
    
    # Merge in target variable
    df_output_array_full = pd.concat([dataframe, df_output_array], axis=1)
    
    print(f'Shape of final dataframe: {df_output_array_full.shape}')
    
    return df_output_array_full