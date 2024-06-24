<div align="center">
  <h1>AUTO LABELING</h1>

  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">

  <img src="https://img.shields.io/badge/LLM-phi3%20mini%204k%20onnx-lightblue?style=for-the-badge&logo=phi3" alt="Phi-3-mini-4k-instruct-onnx">

  <img src="https://img.shields.io/badge/Model-GroundingDINO-orange?style=for-the-badge" alt="GroundingDINO">

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

```mermaid
classDiagram
    class Dataset {
        - parent_path: str
        + images_dir(path: str) : list[str]
        + save_data(data: dict) : json
        + load_data(data: json) : dict
        + add_alias(alias: alias)
        + add_image(image: Image)
    }

    class Image {
        - image_absolute_path: str
        + load(load_type) : None
        + load_gt(path: str) : list[tuple[int, list[float]]]
    }

    class LLM {
        - performance_dictionary: dict
        + get_alias() : alias
    }

    class modello_vis {
        - img: ['PIL', np.array]
        - text: str
        + predict(img) : results
    }

    class Processing {
        + Confusion_Matrix(results: list, gt: list[tuple[int, list[float]]]) : list[classe_id]
    }

    class metrics {
        - list[tp, fp, fn]
        + get_metrics() : list
    }

    class Alias {
        - alias
        - metrics
        + get_aliases() : dict
    }

    Dataset --> Image : add_image
    Dataset --> LLM : add_alias
    Image --> modello_vis
    LLM --> modello_vis : get_alias
    modello_vis --> Processing : predict
    Processing --> metrics : Confusion_Matrix
    metrics --> Alias : get_metrics
    Alias --> Dataset : get_aliases
```

## :clipboard: Requirements

Make shure to have 'huggingface-cli' installed on your machine, or download it from [here](https://huggingface.co/docs/huggingface_hub/guides/cli).

To install the libraries used in the programs:

Make shure to have started your python virtual environment!
In order to keep your python interpreter clean on your machine.

On Linux:

```bash
$/auto_labeling> source .venv/bin/activate
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
