from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AlbertTokenizer, AlbertForSequenceClassification, RobertaTokenizer,RobertaForSequenceClassification
from transformers import TrainingArguments
from transformers import TrainingArguments, Trainer, AutoTokenizer
import numpy as np
import torch

#REPLACE WITH YOUR PATH...........
scores_json_file = 

#PRETRAINED LANGUAGE MODEL...........
pretrained_model="albert-xxlarge-v2"

data_files = {"train": scores_json_file}

dataset = load_dataset("json",data_files=data_files)
print(dataset)
print("")
print(dataset["train"][30])

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

def tokenize_function(examples):
    return tokenizer(examples["sentence"],truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("")
print(tokenized_dataset)
print("")


model = AlbertForSequenceClassification.from_pretrained(pretrained_model,
                                                        num_labels=77,
                                                        ignore_mismatched_sizes=True)

training_args = TrainingArguments(output_dir="/gaueko0/users/dromero012/Fine-Tuning_Tests/Version_1/Output_Model",
                                  per_device_train_batch_size=4,
                                  do_train=True,
                                  do_eval=True,
                                  do_predict=False,
                                  learning_rate=5e-05,
                                  weight_decay=0.01,
                                  adam_beta1=0.9,
                                  adam_beta2=0.98,
                                  adam_epsilon=1e-6,
                                  max_steps=5000,
                                  logging_strategy="steps",
                                  gradient_accumulation_steps=4,
                                  logging_steps=100,
                                  logging_dir="/gaueko0/users/dromero012/Fine-Tuning_Tests/Version_1/logs",
                                  report_to="tensorboard",
                                  seed=96,
                                  data_seed=96,
                                  eval_steps=100,
                                  evaluation_strategy="steps",
                                  dataloader_drop_last=False,
                                  fp16=True)

print("")
print(training_args)
print("")

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    comb=torch.from_numpy(logits)
    t = 2
    out = torch.softmax(comb.float()/t, dim=1)
    predictions = np.argmax(out, axis=1)
    labels = np.argmax(labels,axis=1)

    return metric.compute(predictions=predictions, references=labels)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        temperature=2
        labels = inputs.get("labels")
       
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        predictions = torch.log_softmax(logits / temperature, dim=1)

        loss=torch.nn.functional.kl_div(predictions, labels, reduction='sum') * (temperature ** 2) / predictions.shape[0]

        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['train'],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)


trainer.train()
print(trainer.predict(tokenized_dataset['train']))

var=trainer.predict(tokenized_dataset['train'])
print(var[0])

comb=torch.from_numpy(var[0])
t = 2
out = torch.softmax(comb.float()/t, dim=1)
print(out)

np.savetxt('output.txt',out)
