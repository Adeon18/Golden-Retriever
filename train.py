import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer, \
    SentenceTransformerTrainingArguments

import fire
from huggingface_hub import hf_hub_download


def train(model_name: str = "intfloat/multilingual-e5-small", ):
    train_parquet = "articles_10k_train.parquet"
    test_parquet = "articles_10k_test.parquet"

    print(f'Downloading training dataset')

    hf_hub_download(repo_id="m-rudko-pn/ukrainian-wikipedia-articles", filename=train_parquet, repo_type="dataset")
    hf_hub_download(repo_id="m-rudko-pn/ukrainian-wikipedia-articles", filename=test_parquet, repo_type="dataset")

    print(f'\nDownload completed, reading parquet files')

    train_df = pd.read_parquet(train_parquet)[["section_title", "text"]].rename(columns={
        "section_title": "anchor",
        "text": "positive"
    })

    print(f'Read training dataset with {len(train_df)} samples')

    test_df = pd.read_parquet(test_parquet)[["section_title", "text"]].rename(columns={
        "section_title": "anchor",
        "text": "positive"
    })

    print(f'Read test dataset with {len(test_df)} samples')

    train_dataset = Dataset.from_pandas(train_df)

    test_dataset = Dataset.from_pandas(test_df)

    args = SentenceTransformerTrainingArguments(
        output_dir="output",
        num_train_epochs=12,
        per_device_train_batch_size=48,
        gradient_accumulation_steps=10,
        warmup_steps=100,
        logging_steps=40,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="steps",
        eval_strategy="steps",
        bf16=True
    )

    model = SentenceTransformer(model_name)

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss=losses.MultipleNegativesRankingLoss(model=model),
        args=args
    )

    print(f'Begin training model {model_name}')

    trainer.train()


if __name__ == "__main__":
    fire.Fire(train)
