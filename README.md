<div align="center">
  <h1>AUTO LABELING</h1>

  <img src="https://img.shields.io/badge/LLM-phi3%20mini%204k%20onnx-lightblue?style=for-the-badge&logo=phi3" alt="Phi-3-mini-4k-instruct-onnx">

  <br>

  <img src="https://img.shields.io/badge/Spellcheck-Pass-green?style=flat" alt="Spellcheck Pass">

  <img src="https://img.shields.io/badge/OS%20platform%20supported-Windows-green?style=flat" alt="OS platform supported Windows">

  <img src="https://img.shields.io/badge/OS%20platform%20supported-Unix-green?style=flat" alt="OS platform supported Unix">

  <img src="https://img.shields.io/badge/Language-Python-yellowgreen?style=flat" alt="Language Python">

  <img src="https://img.shields.io/badge/PEP8%20CheckOnline-Passing-green" alt="PEP8 CheckOnline Passing">

  <img src="https://img.shields.io/badge/Test-Pass-green" alt="Test Pass">
</div>

## Table of Contents

- [Description](#pencil2-description)
- [Requirements](#clipboard-requirements)
- [Execution](#diamond_shape_with_a_dot_inside-execution)
- [References](#link-references)
- [Authors](#busts_in_silhouette-authors)

## :pencil2: Description

UML of the project:

![Image](data/img_readme/SchemaLogicoPipeline.drawio.jpg)

## :clipboard: Requirements

To install the requiments:

Make shure to have started your python virtual environment!
In order to keep your python interpreter clean on your machine.

On Linux:

```bash
$/auto_labeling> souce .venv/bin/activate
```

On windows:

```bash
$/auto_labeling> .venv/Scripts/activate
```

Then procede with the installation of the libraries used in the packet

```bash
$/auto_labeling> pip install -r requirements.txt
```

## :diamond_shape_with_a_dot_inside: Execution

```bash
$/auto_labeling/bin> python pipeline.py
```

## :link: References

- [Grounding-DINO](https://huggingface.co/IDEA-Research/grounding-dino-base)
- [PHI-3 model CPU](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi-3-tutorial.md#run-on-cpu)
- [PHI-3 model ONNX](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/tree/main/cpu_and_mobile)

## :busts_in_silhouette: Authors

- GioMont3
- Daniele-Cannella
- Gulia-peretti
- Alessandro
- AbitanteDiScampia
- Neetre
