from mteb import MTEB
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata
import sys

class UkrWikiRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
    name="UkrWikiRetrieval",
    dataset={
        "path": "m-rudko-pn/ukrainian-wikipedia-articles",
        "revision": "1",
    },
    description=(
        "Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants "
        + "(including 115 distinct languages and their scripts)"
    ),
    type="Retrieval",
    category="s2p",
    modalities=["text"],
    eval_splits=["test"],
    eval_langs=["ukr"],
    main_score="ndcg_at_10",
    )
    
    def load_data(self, **kwargs):
        # split = kwargs.get("split", "test[:100%]")

        dataset = load_dataset("m-rudko-pn/ukrainian-wikipedia-articles", split="test[:100%]")
        
        corpus = {}
        queries = {}
        relevant_docs = {}

        title_to_ids = {}

        for i, example in enumerate(dataset):
            doc_id = str(example["page_id"])
            title = example["section_title"]
            text = example["text"]

            corpus[doc_id] = {
                "title": "",
                "text": text if text is not None else "",
            }

            if title not in title_to_ids:
                title_to_ids[title] = []
            title_to_ids[title].append(doc_id)

        for title, doc_ids in title_to_ids.items():
            for doc_id in doc_ids:
                queries[doc_id] = title if title is not None else ""
                relevant_docs[doc_id] = {rel_id: 1 for rel_id in doc_ids}

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}

        print(f"Loaded {len(self.corpus['test'])} documents.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmarks.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    model = SentenceTransformer(model_name)
    benchmark = MTEB(tasks=[UkrWikiRetrieval()])
    res = benchmark.run(model, verbosity=2, output_folder="results/", encode_kwargs={"batch_size": 50})