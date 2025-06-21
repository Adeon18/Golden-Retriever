from mteb import MTEB
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata
import sys

class UkrPravdaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
    name="UkrPravdaRetrieval",
    dataset={
        "path": "shamotskyi/ukr_pravda_2y",
        "revision": "be2be302c4c659d362b6fae64d7526cd1901bca6",
    },
    description=(
        "Ukrainian Pravda Articles Retrieval Task."
    ),
    type="Retrieval",
    category="s2p",
    modalities=["text"],
    eval_splits=["train"],
    eval_langs=["ukr"],
    main_score="ndcg_at_10",
    )
    
    def load_data(self, **kwargs):
        dataset = load_dataset("shamotskyi/ukr_pravda_2y")["train"]
    
        corpus = {}
        queries = {}
        relevant_docs = {}

        title_to_ids = {}

        for i, example in enumerate(dataset):
            doc_id = str(example["art_id"])
            title = example["ukr_title"]
            text = example["ukr_text"]

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

        self.corpus = {"train": corpus}
        self.queries = {"train": queries}
        self.relevant_docs = {"train": relevant_docs}

        print(f"Loaded {len(self.corpus['train'])} documents.")


class UkrWikiRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
    name="UkrWikiRetrieval",
    dataset={
        "path": "m-rudko-pn/ukrainian-wikipedia-articles",
        "revision": "2f6a643539579c8a1f8358b71cb0e5f3c57673cc",
    },
    description=(
        "Ukrainian Wikipedia Articles Retrieval Task. "
        "This task involves retrieving relevant articles from the Ukrainian Wikipedia based on given queries."
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
    benchmarks = {
        "UkrWikiRetrieval": UkrWikiRetrieval,
        "UkrPravdaRetrieval": UkrPravdaRetrieval,
    }

    if len(sys.argv) < 3:
        print("Usage: python benchmarks.py <model_name> <benchmark_name>")
        print("Available benchmarks:", ", ".join(benchmarks.keys()))
        sys.exit(1)

    model_name = sys.argv[1]
    benchmark_name = sys.argv[2]

    if benchmark_name not in benchmarks:
        print(f"Unknown benchmark: {benchmark_name}")
        print("Available benchmarks:", ", ".join(benchmarks.keys()))
        sys.exit(1)

    benchmark_class = benchmarks[benchmark_name]

    model_name = sys.argv[1]

    model = SentenceTransformer(model_name)
    benchmark = MTEB(tasks=[benchmark_class()])
    res = benchmark.run(model, verbosity=2, output_folder="results/", encode_kwargs={"batch_size": 50})