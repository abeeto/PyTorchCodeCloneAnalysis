# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data Preparation
#
# This notebook prepares train/test data for the demo evaluations.

# %% [markdown]
# ## Libraries

# %%
from lenskit import crossfold
from lenskit.datasets import MovieLens, ML100K

# %%
import seedbank
seedbank.initialize(20220908)

# %% [markdown]
# ## Data Setup
#
# Let's set up some data sets:

# %%
ml100k = ML100K('data/ml-100k/')
ml20m = MovieLens('data/ml-20m/')

# %% [markdown]
# ## ML-100K train-test
#
# We're going to create a single train-test split for the ML-100K data set.

# %%
train, test = next(crossfold.sample_users(ml100k.ratings, 1, 100, crossfold.SampleN(5)))

# %%
train.to_parquet(ml100k.path / 'train.parquet', index=False, compression='zstd')

# %%
test.to_parquet(ml100k.path / 'test.parquet', index=False, compression='zstd')

# %% [markdown]
# ## ML-20M train-test
#
# And the same for ML-20M.

# %%
train, test = next(crossfold.sample_users(ml20m.ratings, 1, 10000, crossfold.SampleN(5)))

# %%
train.to_parquet(ml20m.path / 'train.parquet', index=False, compression='zstd')

# %%
test.to_parquet(ml20m.path / 'test.parquet', index=False, compression='zstd')

# %%
