"""constellation-quant package installation."""

from pathlib import Path
from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def _read_requirements() -> list[str]:
    req_file = ROOT / "requirements.txt"
    lines = req_file.read_text().splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


setup(
    name="constellation-quant",
    version="0.1.0",
    description="Graph + temporal deep learning for cross-sectional S&P 500 ranking.",
    long_description=(ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="Zahir Nikraftar",
    url="https://github.com/zahirnik/constellation-quant",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(include=["constellation_quant", "constellation_quant.*"]),
    install_requires=_read_requirements(),
    entry_points={
        "console_scripts": [
            "cq-download = scripts.download_data:main",
            "cq-train    = scripts.train:main",
            "cq-ablation = scripts.run_ablation:main",
            "cq-evaluate = scripts.evaluate:main",
            "cq-report   = scripts.generate_report:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
