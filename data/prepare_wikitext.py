import os, json, urllib.request
import numpy as np
import tiktoken

URLS = {
    "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "val":   "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
}

def main():
    os.makedirs("wikitext", exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    for split, url in URLS.items():
        print(f"[{split}] 다운로드 중: {url}")
        urllib.request.urlretrieve(url, f"wikitext/{split}.txt")
        with open(f"wikitext/{split}.txt", encoding="utf-8") as f:
            text = f.read()
        tokens = enc.encode_ordinary(text)
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(f"wikitext/{split}.bin")
        print(f"  -> {len(arr):,} tokens 저장 완료")

    meta = {"vocab_size": 50257, "tokenizer": "gpt2"}
    with open("wikitext/meta.json", "w") as f:
        json.dump(meta, f)
    print("완료!")

if __name__ == "__main__":
    main()
