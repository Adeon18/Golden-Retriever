# Golden-Retriever

## Set up

This project uses `uv` package manager. To install it, follow the steps
described [here](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).

To install the dependencies, run the following command:

```bash
uv sync
```


## Clear notebooks on push

Notebooks take up a lot of space when storing image outputs. To avoid this, the notebooks can be automatically cleared
before on every commit.

To achieve this, you have to specify .githooks directory to use provided git hooks. Run the following command:

```bash
git config core.hooksPath .githooks
```

Now, hooks from .githooks/ are used rather than .git/hooks.