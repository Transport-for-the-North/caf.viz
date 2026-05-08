<div align="center" style="background-color: white;">
<a href="https://www.transportforthenorth.com/">
<img src="https://www.transportforthenorth.com/wp-content/themes/tfn-theme/img/logo.svg"
  alt="Transport for the North logo">
</a>
</div>

<h1 align="center">CAF.viz</h1>

<p align="center">
<a href="https://transport-for-the-north.github.io/CAF-Handbook/python_tools/framework.html">
  <img alt="CAF Status - Pre-Alpha" src="https://img.shields.io/badge/CAF%20Status-Pre--Alpha-orange">
</a>
</p>
<p align="center">
<a href="https://pypi.org/project/caf.viz/">
  <img alt="Supported Python versions" src="https://img.shields.io/pypi/pyversions/caf.viz.svg?style=flat-square">
</a>
<a href="https://pypi.org/project/caf.viz/">
  <img alt="Latest release" src="https://img.shields.io/github/release/transport-for-the-north/caf.viz.svg?style=flat-square&maxAge=86400">
</a>
<a href="https://anaconda.org/conda-forge/caf.viz">
  <img alt="Conda" src="https://img.shields.io/conda/v/conda-forge/caf.viz?style=flat-square&logo=condaforge">
</a>
</p>
<p align="center">
<a href="https://github.com/transport-for-the-north/caf.viz/actions?query=event%3Apush">
  <img alt="Testing Badge" src="https://img.shields.io/github/actions/workflow/status/transport-for-the-north/caf.viz/tests.yml?style=flat-square&logo=GitHub&label=Tests">
</a>
<a href="https://app.codecov.io/gh/transport-for-the-north/caf.viz">
  <img alt="Coverage" src="https://img.shields.io/codecov/c/github/transport-for-the-north/caf.viz.svg?branch=main&style=flat-square&logo=CodeCov">
</a>
<a href='https://cafviz.readthedocs.io/en/stable/'>
  <img alt='Documentation Status' src="https://img.shields.io/readthedocs/cafviz?style=flat-square&logo=readthedocs">
</a>
</p>

> [!WARNING]  
> This package is in an early stage of development so features may change or be removed.
> If using this package it is recommended to set a specific version and check before
> upgrading to a new version.

Visualisation functionality and tools for transport related data.

> [!TIP]
> For more detailed information including a user guide, tutorials and API reference see the full
> [caf.viz documentation](https://cafviz.readthedocs.io/en/stable/)

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
  - [What does it do?](#what-does-it-do)
  - [Main Features](#main-features)
    - [Work-in-Progress](#work-in-progress)
  - [Who is it for?](#who-is-it-for)
- [Where to get it](#where-to-get-it)
  - [Installation from GitHub](#installation-from-github)
- [Usage](#usage)
  <!-- - [Command Line](#command-line)-->
- [Documentation](#documentation)
- [What is CAF?](#what-is-caf)
- [Contribution](#contribution)
- [Contact Us](#contact-us)

## Overview

### What does it do?

> [!IMPORTANT]
> This section of the README hasn't been written yet, but it will contain a brief
> description of what the tool is intended to do.

### Main Features

- **Simplified 2D graphs** - wrappers around matplotlib to simplify the creation of
  basic 2D plots (scatter, line and hexbin) and to simplify the creation of figures
  with multiple Axes.
- **TfN Stylesheet** - custom stylesheet for matplotlib using TfN's colour scheme.

#### Work-in-Progress

- **Static Heatmaps** - wrapper around GeoPandas and matplotlib to simplify the creation
  of heatmap images (see #24).

> [!WARNING]
> These features are work-in-progress and are not available in a released version of caf.viz, to
> access these features a specific branch of caf.viz should be installed, see [Installation from GitHub](#installation-from-github).

### Who is it for?

- **Target audience:** GIS Analysts, Transport Analysts, Data Analysts
- **CAF Analytical Stage:** Analysis, Appraisal, Evaluation

![CAF Analytical Process Diagram](https://github.com/Transport-for-the-North/.github/blob/21a428e81880639839e221940881572cdee24d5a/profile/ProcessDiagram.png?raw=true)

For more details on CAF Analytical Stages see the [description within TfN's GitHub homepage](https://github.com/Transport-for-the-North)

## Where to get it

The latest released version are available at the
[Python Package Index (PyPI)](https://pypi.org/project/caf.viz) and on
[Conda](https://anaconda.org/conda-forge/caf.viz).

```sh
conda install -c conda-forge caf.viz
```

```sh
pip install caf.viz
```

> [!TIP]
>
> - See the [Quick Start Guide](https://cafviz.readthedocs.io/en/stable/start.html#quick-start) for more detailed instructions.
> - See the [requirements.txt](requirements.txt) for the full list of package dependencies.

### Installation from GitHub

> [!WARNING]
> Unreleased GitHub versions should **not** be considered stable.

The latest, unreleased, version can be installed directly from GitHub using:

```sh
pip install "git+https://github.com/transport-for-the-north/caf.viz"
```

> [!TIP]
> `pip install` can install a specific tag, or branch, using `@{tag-name}`
> after the git URL.

## Usage

CAF.viz doesn't currently provide a CLI or GUI so should be called from Python to use,
see the [Tool Usage section](https://cafviz.readthedocs.io/en/stable/usage/index.html)
of the user guide for more details.

<!--
### Command Line

The tool can be run from command line, with the command:

```sh
caf.viz
```

See [Command-Line Interface (User Guide)](https://cafviz.readthedocs.io/en/stable/usage/cli.html)
for full explanations of the parameters.
-->

## Documentation

The code documentation is hosted at <https://cafviz.readthedocs.io/en/stable/>.

## What is CAF?

This tool is part of TfN's [Common Analytical Framework (CAF)](https://github.com/Transport-for-the-North).
CAF is Transport for the North's structured suite of analytical tools designed to support transport
modelling, appraisal, and strategic decision-making.

More information on CAF and details on other CAF tools can be found on [TfN's GitHub Homepage](https://github.com/Transport-for-the-North).

## Contribution

We encourage use of, and contributions to, the repositories within this organisation, licenses are provided within
the repositories and the [organisation contribution guide](https://github.com/Transport-for-the-North/.github/blob/main/CONTRIBUTING.rst)
provides details for contributions.

---

## Contact Us

For further information about using this tool or CAF tools in your projects and work contact Transport for the North - <TfNOffer@transportforthenorth.com>

---

[Go to Top](#table-of-contents)
