from __future__ import annotations

import glob
import os.path as osp
import re

import setuptools

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()


def find_version(project_dir=None):
    if not project_dir:
        project_dir = osp.dirname(osp.abspath(__file__))
    file_path = osp.join(
        project_dir,
        "train_app",
        "version.py",
    )
    with open(file_path) as version_file:
        version_text = version_file.read()
    # PEP440:
    # https://www.python.org/dev/peps/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions
    pep_regex = r"([1-9]\d*!)?(0|[1-9]\d*)(\.(0|[1-9]\d*))*((a|b|rc)(0|[1-9]\d*))?(\.post(0|[1-9]\d*))?(\.dev(0|[1-9]\d*))?"
    version_regex = r"VERSION\s*=\s*.(" + pep_regex + ")."
    match = re.match(version_regex, version_text)
    if not match:
        raise RuntimeError("Failed to find version string in '%s'" % file_path)

    version = version_text[match.start(1) : match.end(1)]  # noqa: E203
    return version


def get_dirs(dir_names: list):
    file_names = []  # type: list[str]
    for dir_name in dir_names:
        file_names += glob.glob(dir_name + "/**/*.*")
        file_names += glob.glob(dir_name + "/*.*")
    return file_names


setuptools.setup(
    name="train_app",
    version=find_version(),
    author="ordulu",
    description="Trainer Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.ordulu.com/ai-team/falcon-segmentation-trainer.git",
    project_urls={"Bug Tracker": "https://gitlab.ordulu.com/ai-team/falcon-segmentation-trainer/-/issues"},
    license="Ordulu License",
    packages=setuptools.find_packages(
        include=[
            "train_app",
            "train_app.models",
            "train_app.scripts",
            "train_app.loss",
        ],
    ),
    # exclude_package_data={"train_app": ["__pycache__/*"]},
    package_data={"train_app.scripts": ["config/*"]},
    data_files=[("train_app", get_dirs(["train_app/models", "train_app/scripts/config"]))],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "train_app=train_app.scripts.cli:main",
        ],
    },
)
