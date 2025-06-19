import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer


train_df = pd.read_parquet("articles_10k_train.parquet")[["section_title", "text"]].rename(columns={
    "section_title": "anchor",
    "text": "positive"
})

test_df = pd.read_parquet("articles_10k_test.parquet")[["section_title", "text"]].rename(columns={
    "section_title": "anchor",
    "text": "positive"
})

train_dataset = Dataset.from_pandas(train_df)

test_dataset = Dataset.from_pandas(test_df)

from sentence_transformers import losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments

args = SentenceTransformerTrainingArguments(
    output_dir="output",
    num_train_epochs=12,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=10,
    warmup_steps=100,
    logging_steps=40,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy="steps",
    eval_strategy="steps",
    bf16=True
)

model = SentenceTransformer(
    # "all-MiniLM-L6-v2"
    # "all-distilroberta-v1"
    "intfloat/multilingual-e5-base",
    # "intfloat/multilingual-e5-small",
)

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=losses.MultipleNegativesRankingLoss(model=model),
    args=args
)

trainer.train()