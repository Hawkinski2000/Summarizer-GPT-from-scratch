from datasets import load_dataset
import tiktoken
import os
import numpy as np


ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")

enc = tiktoken.get_encoding("gpt2")

# ----------------------------------------------------------------------------

# Calculate average tokens per article:

# num_articles = len(ds["article"])
# mean = 0
# for article in ds["article"]:
#     mean += len(enc.encode_ordinary(article)) / num_articles

# print(mean) # Average article is ~868 tokens, so use a block size of 1024

# ----------------------------------------------------------------------------

# Calculate average tokens per summary:

# num_summaries = len(ds["highlights"])
# mean = 0
# for summary in ds["highlights"]:
#     mean += len(enc.encode_ordinary(summary)) / num_summaries

# print(mean) # Average summary is ~66 tokens, so use a block size of 128

# ----------------------------------------------------------------------------

# Create padded arrays of tokenized articles and summaries:

eot = enc._special_tokens['<|endoftext|>'] # End of text token
pad_token = enc.encode_ordinary('0')[0] # Pad token

articles = []
summaries = []

# Articles:

for article in ds["article"]:
    tokens = enc.encode_ordinary(article)
    
    # Pad or truncate articles so they're exactly 1024 tokens:

    if len(tokens) < 1023: # Pad with token 0 if shorter than 1023
        enc_article = [eot] + tokens + [pad_token] * (1023 - len(tokens))
    else: # Truncate if longer than 1023
        enc_article = [eot] + tokens[:1023]

    articles += enc_article

# Summaries:

for summary in ds["highlights"]:
    tokens = enc.encode_ordinary(summary)
    
    # Pad or truncate summaries so they're exactly 128 tokens:

    if len(tokens) < 127: # Pad with token 0 if shorter than 127
        enc_summary = [eot] + tokens + [pad_token] * (127 - len(tokens))
    else: # Truncate if longer than 127
        enc_summary = [eot] + tokens[:127]

    summaries += enc_summary

# ----------------------------------------------------------------------------

# Create a "cnn_dailymail" folder and save articles and summaries as .npy

local_dir = "cnn_dailymail"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Save articles:

articles_np = np.array(articles)
assert (
    (0 <= articles_np).all() and
    (articles_np < 2 ** 16).all()
    ), "token dictionary too large for uint16"
articles_np_uint16 = articles_np.astype(np.uint16)
articles_path = os.path.join(DATA_CACHE_DIR, "articles.npy")
np.save(articles_path, articles_np_uint16)

# Save summaries:

summaries_np = np.array(summaries)
assert (
    (0 <= summaries_np).all() and
    (summaries_np < 2 ** 16).all()
    ), "token dictionary too large for uint16"
summaries_np_uint16 = summaries_np.astype(np.uint16)
summaries_path = os.path.join(DATA_CACHE_DIR, "summaries.npy")
np.save(summaries_path, summaries_np_uint16)
