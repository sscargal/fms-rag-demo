#!/usr/bin/env python
import subprocess
import os

DATA_DIR = "/data/jasonc/"
if not os.path.exists(f"{DATA_DIR}enwiki-latest-pages-articles-multistream.xml.bz2"):
    print("Dump file not found. Downloading dump")
    subprocess.run(["wget", "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2", 
                    "-P", DATA_DIR])
else:
    print("Dump file already found on disk. Skipping to extracting from dump")

SUBSET_SIZE = "1G"
subprocess.run(["python", "-m", "wikiextractor.WikiExtractor", f"{DATA_DIR}enwiki-latest-pages-articles-multistream.xml.bz2",
                "--output", f"{DATA_DIR}extracted_text/",
                "--templates", f"{DATA_DIR}enwiki_dump_templates",
                "--bytes", SUBSET_SIZE])