"""
VOC-ArchNLP v1.0.0 — setup.py
Hak Cipta (Program Komputer) — Universitas Bhinneka Nusantara, 2026
Pencipta: Mukhlis Amien
"""

from setuptools import setup, find_packages

setup(
    name="voc-archnlp",
    version="1.0.0",
    author="Mukhlis Amien",
    author_email="amien@ubhinus.ac.id",
    description=(
        "NLP pipeline for mining archaeological information from Dutch VOC colonial archives"
    ),
    long_description=open("../../../docs/HKI/DESKRIPSI_PROGRAM.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="CC BY 4.0",
    url="https://github.com/mukhlisamin/volcarch",
    packages=find_packages(where=".."),
    package_dir={"": ".."},
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.28",
        "pathlib",
    ],
    extras_require={
        "nlp": [
            "spacy>=3.5",
            "transformers>=4.35",
            "torch>=2.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "voc-archnlp=voc_archnlp.cli:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords=[
        "NLP", "VOC", "colonial Dutch", "archaeology", "Indonesia",
        "historical text mining", "GLOBALISE", "digital humanities"
    ],
)
